import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset

class BiasedDataset(ConfounderDataset):
    def __init__(self, data, subset=None, transform=None):
        self.data = data
        #self.data['name'] = np.arange(len(self.data['y']), dtype=np.int)
        self.n_classes = 2
        self.subset = subset
        self.images = self.data['X']
        self.targets = self.data['y']
        self.names = np.arange(len(self.data['y']), dtype=np.int)
        try:
            self.masks = self.data['masks']
        except:
            self.masks = self.names
            
        try:
            self.groups = self.data['g']
        except:
            self.groups = self.names
            
        assert len(self.names) == len(self.targets)
         
        if self.subset is not None:
            self.images = self.images[self.subset]
            self.targets = self.targets[self.subset]
            self.names = self.names[self.subset]
            try:
                self.groups = self.groups[self.subset]
            except:
                self.groups = []
        # This is the setting for normal Biased (2 environments)
        self.confounder_names = ['bias_scheme']
        #self.confounder_names = ['bottom', 'top', 'left', 'right']
        
        if transform is None: 
            self.transform = get_transform_skin(train=True)
        else:
            self.transform = transform
        
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        # Map to groups
        # This is the setting for normal Biased (2 envs)
        self.n_groups = self.n_classes * pow(2, len(self.confounder_names))
        #self.n_groups = self.n_classes * len(self.confounder_names)
        self.group_array = (self.targets*(self.n_groups/2) + self.groups).astype('int')
        self.n_confounders = len(self.confounder_names)
        self.targets = torch.LongTensor(self.targets)
        try:
            self.group_array = torch.LongTensor(self.group_array)
        except:
            self.group_array = []

    def __getitem__(self, index):
        x = self.images[index]
        y = self.targets[index]
        #name = self.names[index]
        if len(self.groups) > 0:
            g = self.group_array[index]
        else:
            g = self.group_array
        
        
        if self.transform:
            #x = Image.fromarray(self.images[index].astype(np.uint8))
            #x = self.transform(x)
            try:
                x = Image.fromarray(x.astype(np.uint8)).convert('RGB')
                x = self.transform(x)
            except:
                
                x = self.transform(image=x.astype(np.uint8))["image"]
            
        
        return x, y, g
    
    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        group_name = f'label = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name
    
    def __len__(self):
        return len(self.targets)


def get_transform_skin(train):
    if train:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    return transform

class BiasedDatasetWithMask(BiasedDataset):
    """
    BiasedDataset that also returns masks.
    """

    def __getitem__(self, index):
        if len(self.masks) > 0:
            m = self.masks[index]
            m = Image.fromarray(m.astype(np.uint8)).convert('L')
            mask = get_transform_mask(m).cpu().data.numpy()
            mask = np.squeeze(mask)
            mask[mask>0.0] = 1.0
            mask[mask<=0.0] = 0.0
        else:
            mask = []
        return super().__getitem__(index), mask


def get_transform_mask(mask):

    transform_mask = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.0], [1.0])
    ])

    return transform_mask(mask)

