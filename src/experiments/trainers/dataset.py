import glob, os
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, RandRotate90d, RandFlipd, Resized, RandAffined

class NonToContrastDataset(Dataset):
    def __init__(self, path_in='', files=[], transform=None):
        self.transform = transform
        self.pairs = []
        for x in files:
            self.pairs.extend([
                (a, b) 
                for a,b in zip(
                    sorted(glob.glob(os.path.join(f"{path_in}/{x}/", "nat_*"))), 
                    sorted(glob.glob(os.path.join(f"{path_in}/{x}/", "art_*")))
                ) 
            ])
        self.pairs = self.pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        file = {
            'native': torch.from_numpy(np.load(self.pairs[idx][0]).clip(-1000., 1000.)/1000.)[None]+1.,  
            'contrast': torch.from_numpy(np.load(self.pairs[idx][1]).clip(-1000., 1000.)/1000.)[None]+1.
        }

        if self.transform:
            file = self.transform(file)

        # We need this strange move with +1. and -1., because of zeros padding in monai Affined
        return (file['native']-1.).clip(-1., 1.).as_tensor(), (file['contrast']-1.).clip(-1., 1.).as_tensor()

def get_transform_test(cfg):
    return Compose([
        Resized(keys=("native", "contrast"), spatial_size=(cfg['H'], cfg['W'])), 
    ])

def get_transform_train(cfg):
    return Compose([
        RandFlipd(keys=["native", "contrast"], prob=cfg['RandFlipd']['prob'], spatial_axis=0),
        RandFlipd(keys=["native", "contrast"], prob=cfg['RandFlipd']['prob'], spatial_axis=1),
        RandRotate90d(keys=["native", "contrast"], prob=cfg['RandRotate90d']['prob'], max_k=cfg['RandRotate90d']['max_k']),
        RandAffined(
            keys=["native", "contrast"],
            prob=cfg['RandAffined']['prob'],
            rotate_range=(cfg['RandAffined']['rotate_range']/180*np.pi,),       
            translate_range=(cfg['RandAffined']['translate_range'][0] * cfg['H'], cfg['RandAffined']['translate_range'][1] * cfg['W']),
            shear_range=(cfg['RandAffined']['shear_range']/180*np.pi,),
            mode="bilinear",
            padding_mode="zeros",
        ),
        Resized(keys=("native", "contrast"), spatial_size=(cfg['H'], cfg['W'])), 
    ])