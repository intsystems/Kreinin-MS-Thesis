import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import yaml
import wandb
from tqdm import tqdm
import json

from utils import get_lr, compute_alpha, get_beta_schedule
from datasest import NonToContrastDataset, get_transform_test, get_transform_train


class DiffusionTrainer:
    def __init__(self, config_name=''):
        self.set_configs(config_name)
        self.set_wandb()
        self.set_net()
        self.set_opt_sched()
        self.set_loaders()
        self.best_acc = 1e8 
    
    def criterion(self, x_image, x_ground, t, eps, beta, config):
        a = (1-beta).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = x_ground * a.sqrt() + eps * (1.0 - a).sqrt()
        output = self.net(torch.cat([x_image, x], dim=1), t.float())
        
        loss = (eps - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
        config['loss'].append(loss.item())
        
        return loss

    def get_time(self, n):
        skip = self.num_timesteps // self.config['timesteps']
        t_intervals = torch.arange(-1, self.num_timesteps, skip)
        t_intervals[0] = 0
        
        idx_1 = torch.randint(0, len(t_intervals), size=(n // 2 + 1,))
        idx_2 = len(t_intervals)-idx_1-1
        idx = torch.cat([idx_1, idx_2], dim=0)[:n]
        t = t_intervals[idx].to(self.device)
        
        return t

    @torch.no_grad()
    def test(self, i_epoch):
        # Test
        self.net.eval()
        config_log = self.get_log_config()
        
        # Inference
        loop = tqdm(enumerate(self.test_loader), total=len(self.test_loader), leave=False)
        for batch_idx, (native, contrast) in loop:
            native, contrast = native.to(self.device), contrast.to(self.device)
            loss = self.criterion(contrast, native, self.get_time(native.shape[0]), torch.randn_like(native), self.betas, config_log)
            
            # LOOPA and PUPA
            loop.set_description(f"Epoch (Test)[{i_epoch}/{self.num_epochs}]")
            loop.set_postfix(loss=np.mean(config_log['loss']))

        config_log['i_epoch'] = i_epoch
        if self.wandb_use:
            self.wandb_log_nums(config_log, name='Test')

        # Save checkpoint.
        self.model_selection(config_log)

    def train(self, i_epoch):
        config_log = self.get_log_config()
        self.net.train()
        # Train
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
        for batch_idx, (native, contrast) in loop:
            native, contrast = native.to(self.device), contrast.to(self.device)

            loss = self.criterion(contrast, native, self.get_time(native.shape[0]), torch.randn_like(native), self.betas, config_log)
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # LOOPA and PUPA
            loop.set_description(f"Epoch (Train)[{i_epoch}/{self.num_epochs}]")
            loop.set_postfix(loss=np.mean(config_log['loss']))

        config_log['i_epoch'] = i_epoch
        if self.wandb_use:
            self.wandb_log_nums(config_log, name='Train')

    def model_selection(self, metrics):
        if np.mean(metrics['loss']) < self.best_acc:
            self.best_acc = np.mean(metrics['loss'])
            self.save_model(metrics, metrics['i_epoch'], self.model_name + f'_best')
        else:
            self.save_model(metrics, metrics['i_epoch'], self.model_name + f'_last')

    def save_model(self, metrics, i_epoch, name):
        for key in metrics.keys():
            if isinstance(metrics[key], list):
                metrics[key] = np.mean(metrics[key])

        state = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.save_config,
            'epoch': i_epoch,
            'acc': self.best_acc
        }

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{name}.pth')
        print(f'Saved!:) Epoch[{i_epoch}], loss = {np.mean(metrics["loss"]):.04f}')


    def fit(self):
        for i_epoch in range(self.num_epochs):
            self.train(i_epoch)
            self.test(i_epoch)
        if self.wandb_use:
            self.run.finish()

    def set_net(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.config['device'])
        self.net = get_net(self.config['net_name'], self.config_net)
        self.net = self.net.to(self.device)

        betas = get_beta_schedule(
            beta_start=self.config['beta_start'],
            beta_end=self.config['beta_end'],
            num_diffusion_timesteps=self.config['num_diffusion_timesteps'],
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        if self.wandb_use:
            wandb.watch(self.net, log_freq=100)

        self.grad_clip = self.config.get('grad_clip', 1.0)

    def set_loaders(self):
        files_train, files_test = self.get_files()
        transform_test = get_transform_test(self.save_config['transform_test'])
        transform_train = get_transform_train(self.save_config['transform_train'])
        
        train_ds = NonToContrastDataset(
            path_in=self.save_config['path_in'],
            files=files_train,
            transform=transform_train
        )
        self.train_loader = DataLoader(
            dataset=train_ds,
            shuffle=True,
            num_workers=self.config['num_workers_train'],
            batch_size=self.config['batch_size_train']
        )

        test_ds = NonToContrastDataset(
            path_in=self.save_config['path_in'],
            files=files_test,
            transform=transform_test
        )
        self.test_loader = DataLoader(
            dataset=test_ds,
            shuffle=False,
            num_workers=self.config['num_workers_test'],
            batch_size=self.config['batch_size_test']
        )

    def get_files(self):
        files = torch.load(self.files_names, weights_only=False)
        return files['train'], files['test']

    def set_opt_sched(self):
        self.optimizer = torch.optim.Adam(
            params=self.net.parameters(),
            lr=self.config_optimizer['learning_rate'],
            betas=self.config_optimizer['betas'],
            eps=self.config_optimizer['eps'],
            amsgrad=self.config_optimizer['amsgrad'],
            weight_decay=self.config_optimizer['weight_decay']
        )
    
    def load(self, filename, config_name=None):
        checkpoint = torch.load(f'./checkpoint/{filename}.pth', map_location='cpu', weights_only=False)
        self.set_net()
        self.net.load_state_dict(checkpoint['net'])
        self.set_opt_sched()
        self.net = self.net.to(self.device)

        self.best_acc = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']

        if config_name is not None:
            if config_name == 'checkpoint':
                self.set_configs(config=checkpoint['config'])
            else:
                self.set_configs(config_name=config_name)

        if self.wandb_use:
            wandb.watch(self.net, log_freq=100)

    def set_configs(self, config_name='', config=None):
        if config is None:
            with open(f'{config_name}', 'r') as f:
                config = yaml.safe_load(f)
        self.config_net = config['net']
        self.config_optimizer = config['optimizer']
        self.config = config['hyperparams']
        self.files_names = config['files']
        self.save_config = config

    def set_wandb(self):
        self.wandb_use = self.config.get('wandb_use', False)
        if self.wandb_use:
            wandb.login(key=self.config['wandb_key'])
            self.run = wandb.init(project=self.config['project'], config=self.save_config)
            self.model_name = f'run_{self.run.name}_model'
        else:
            self.model_name = 'run_without_wandb_model'
        
        self.log_file = f"./logs/{self.model_name}.json"

        self.start_epoch = 0
        self.num_epochs = self.config['epochs']

    def wandb_log_nums(self, config, name):
        log_data = {
            f'Epoch': config['i_epoch'],
            f'{name} loss': np.mean(config['loss']),
            f'{name} Learning rate': get_lr(self.optimizer),
        }
        
        wandb.log(log_data)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')

    
    def get_log_config(self):
        return {
            'i_epoch': 0,
            'loss': [],
        }