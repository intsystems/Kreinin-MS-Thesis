import torch
import torch.nn as nn
import numpy as np
from UMambaBot_2d import UMambaBot
from monai.networks.nets import SegResNet, UNet, SwinUNETR
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dif_model import DiffusionModel
from flow_matching import TimeResNet

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def get_net(name, cfg):
    if name == 'UNet':
        return UNet(**cfg)
    elif name == 'SegResNet': 
        return SegResNet(**cfg)
    elif name == 'UMambaBot':
        conv_op = convert_dim_to_conv_op(2)
        net = UMambaBot(
            input_channels           = cfg['input_channels'],
            n_stages                 = cfg['n_stages'],
            features_per_stage       = cfg['features_per_stage'],
            conv_op                  = conv_op,
            kernel_sizes             = cfg['kernel_sizes'],
            strides                  = cfg['strides'],
            num_classes              = cfg['num_classes'],
            deep_supervision         = cfg['deep_supervision'],
            n_conv_per_stage         = cfg['n_conv_per_stage'],
            n_conv_per_stage_decoder = cfg['n_conv_per_stage_decoder'],
            conv_bias                = cfg['conv_bias'],
            norm_op                  = get_matching_instancenorm(conv_op),
            norm_op_kwargs           = cfg['norm_op_kwargs'],
            dropout_op               = cfg['dropout_op'],
            dropout_op_kwargs        = cfg['dropout_op_kwargs'],
            nonlin                   = nn.LeakyReLU,
            nonlin_kwargs            = cfg['nonlin_kwargs'],
        )
        net.apply(InitWeights_He(1e-2))
        return net
    elif name == 'SwinUNETR':
        net = SwinUNETR(
            img_size=cfg['img_size'],
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            depths=cfg['depths'],
            num_heads=cfg['num_heads'],
            feature_size=cfg['feature_size'],
            norm_name=cfg['norm_name'],
            normalize=cfg['normalize'],
            spatial_dims=cfg['spatial_dims'],
            downsample=cfg['downsample'],
        )
        return net
    elif name == 'DiffusionModel':
        return DiffusionModel(**cfg)
    elif name == 'FlowMatchingModel':
        return TimeResNet(**cfg)
    
    assert 1==0, f'I do not know about this neural network "{name}"'

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)