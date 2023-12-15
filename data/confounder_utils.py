import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.celebA_dataset import CelebADataset
from data.cub_dataset import CUBDataset
from data.dro_dataset import DRODataset
from data.multinli_dataset import MultiNLIDataset
from data.skin_dataset import SkinDataset
from data.biased_dataset import BiasedDataset
from data.skin_group_dataset import GroupSkinDataset
import pickle
################
### SETTINGS ###
################

confounder_settings = {
    'CelebA':{
        'constructor': CelebADataset
    },
    'CUB':{
        'constructor': CUBDataset
    },
    'MultiNLI':{
        'constructor': MultiNLIDataset
    },
    'Skin':{
        'constructor': SkinDataset
    },
    'Biased':{
        'constructor': BiasedDataset
    },
}

########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args, train, return_full_dataset=False):
    if args.dataset == 'Biased':
        
        if "large" in args.exp_name.lower():
            pickle_dir = '/datasets/abissoto/imagenet-200/'    
        elif "skin" in args.exp_name.lower():
            pickle_dir = '/datasets/abissoto/skin_squared/'

        if "squared" in args.exp_name.lower():
            if args.color_noise is not None:
                exp_type = "_cnoise{}".format(args.color_noise)
            else:
                exp_type = ""

            if args.random_square_position is not None:
                exp_type += "_position"

            if args.square_size is not None:
                exp_type += "_squared{}".format(args.square_size)
            else:
                exp_type += "_squared"
            
        else:
            exp_type = ""

        if  "large" in args.exp_name.lower(): 
            print('loading {}/train{}_{}_{}_bf{}.pickle for train'.format(pickle_dir,exp_type,args.sc1,args.sc2,args.bias_factor))
            train_data_to_dump = pickle.load(open(os.path.join(pickle_dir, 'train' + exp_type + '_{}_{}_bf{}.pickle'.format(args.sc1, args.sc2, args.bias_factor)), 'rb'))
        elif "skin" in args.exp_name.lower():
            print('loading train{}_bf{}.pickle for train'.format(exp_type,args.bias_factor))
            train_data_to_dump = pickle.load(open(os.path.join(pickle_dir, 'train' + exp_type + '_bf{}.pickle'.format(args.bias_factor)), 'rb'))
       
        if args.seed is None:
            rng = np.random.default_rng(1111)
        else:
            rng = np.random.default_rng(args.seed)

        amt_val = int(0.2 * len(train_data_to_dump['y']))
        all_idx = rng.permutation(np.arange(len(train_data_to_dump['y'])))
        val_idx = all_idx[:amt_val]
        train_idx = all_idx[amt_val:]
       
        splits = [train_idx, val_idx, val_idx]

        full_dataset = [confounder_settings[args.dataset]['constructor'](
            data=train_data_to_dump,
            subset=split.astype(int)) for split in splits]

        dro_subsets = [DRODataset(dataset, process_item_fn=None, n_groups=dataset.n_groups,
                                  n_classes=dataset.n_classes, group_str_fn=dataset.group_str) \
                       for dataset in full_dataset]

    else:
        full_dataset = confounder_settings[args.dataset]['constructor'](
            root_dir=args.root_dir,
            target_name=args.target_name,
            confounder_names=args.confounder_names,
            model_type=args.model,
            augment_data=args.augment_data)
        if return_full_dataset:
            return DRODataset(
                full_dataset,
                process_item_fn=None,
                n_groups=full_dataset.n_groups,
                n_classes=full_dataset.n_classes,
                group_str_fn=full_dataset.group_str)

        if train:
            splits = ['train', 'val', 'test']
        else:
            splits = ['test']
        subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
        dro_subsets = [DRODataset(subsets[split], process_item_fn=None, n_groups=full_dataset.n_groups,
                                  n_classes=full_dataset.n_classes, group_str_fn=full_dataset.group_str) \
                       for split in splits]
    return dro_subsets
