import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss import LossComputer
from dataset_loader import CSVDatasetWithName, CSVDataset
#from pytorch_transformers import AdamW, WarmupLinearSchedule
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score, precision_score, recall_score
from data.skin_dataset import get_transform_skin
from data.biased_dataset import BiasedDataset
import pickle

def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None):
    """
    scheduler is only used inside this function if model is bert.
    """

    if is_training:
        model.train()
        if args.model == 'bert':
            model.zero_grad()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    scaler = torch.cuda.amp.GradScaler()
    all_preds = []
    all_labels = []
    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):

            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            with torch.cuda.amp.autocast(): 
                outputs = model(x)
                loss_main = loss_computer.loss(outputs, y, g, is_training)

            all_preds += list(F.softmax(outputs, dim=1).cpu().data.numpy())
            all_labels += list(y.cpu().data.numpy())

            if is_training:
                optimizer.zero_grad()
                scaler.scale(loss_main).backward()
                scaler.step(optimizer)
                scaler.update()

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()


        # Calculate multiclass AUC
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        if args.n_classes == 2:
            auc = roc_auc_score(all_labels, all_preds[:, 1])
        else:
            cm = confusion_matrix(all_labels, all_preds.argmax(axis=1))
            cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            auc = np.trace(cmn) / cmn.shape[0]
            
        print('Epoch: ' + str(epoch) + ' -- AUC: ', str(auc))


        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()
        return auc

def run_eval(epoch, model, loader, args, test_aug=False):
    
    model.eval()
    prog_bar_loader = loader

    all_preds = []
    all_labels = []
    with torch.set_grad_enabled(False):
        for batch_idx, batch in enumerate(prog_bar_loader):
            x, y, _ = batch
            x = x.to("cuda")
            y = y.to("cuda")
            outputs = model(x)
            if not test_aug:
                all_preds += list(F.softmax(outputs, dim=1).cpu().data.numpy())
                all_labels += list(y.cpu().data.numpy())
            else:
                all_preds += list([np.mean(F.softmax(outputs, dim=1).cpu().data.numpy(), axis=0)])
                all_labels += list([y.cpu().data.numpy()[0]])
        # Calculate multiclass AUC
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Removed for multiclass
        auc = roc_auc_score(all_labels, all_preds[:, 1])
        
        
        cm = confusion_matrix(all_labels, all_preds.argmax(axis=1))
        #cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #acc = np.trace(cmn) / cmn.shape[0]
        acc = balanced_accuracy_score(all_labels, all_preds.argmax(axis=1))
        print('confusion matrix:\n', cm)
        #auc = [precision_score(all_labels, all_preds.argmax(axis=1), average=None), recall_score(all_labels, all_preds.argmax(axis=1), average=None)]
        
        print('Epoch: ' + str(epoch) + ' -- AUC: ' + str(auc) + ' -- ACC: ' + str(acc) )
        #print('Epoch: ' + str(epoch) + ' -- ACC: ' + str(acc) )
        #print('[precision,recall]', auc)

        return auc, acc

def train(model, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, fs_observer, wandb, epoch_offset):
    print("fffss observer", fs_observer)
    model = model.cuda()
    CHECKPOINTS_DIR = os.path.join(fs_observer, 'checkpoints')
    BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'model_best')
    LAST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'model_last')
    os.makedirs(CHECKPOINTS_DIR)
    patience_count = 0
    wandb.log({'args':args}, commit=True)

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=0.1,
            patience=5,
            threshold=0.0001,
            min_lr=0,
            eps=1e-08)
    else:
        scheduler = None

    best_val_loss = 100000
    best_val_auc = 0
    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        train_auc = run_epoch(
            epoch, model, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args, 
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler)

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha)
        val_auc = run_epoch(
            epoch, model, optimizer,
            dataset['val_loader'],
            val_loss_computer, 
            logger, val_csv_logger, args,
            is_training=False)

        # Test set; don't print to avoid peeking
        #if dataset['test_data'] is not None:
        #    test_loss_computer = LossComputer(
        #        criterion,
        #        is_robust=args.robust,
        #        dataset=dataset['test_data'],
        #        step_size=args.robust_step_size,
        #        alpha=args.alpha)
        #    run_epoch(
        #        epoch, model, optimizer,
        #        dataset['test_loader'],
        #        test_loss_computer,
        #        None, test_csv_logger, args,
        #        is_training=False)


        metrics_comet = {'epoch':epoch, 'train/loss': train_loss_computer.avg_actual_loss, 'train/max_group_loss': max(train_loss_computer.exp_avg_loss), 'train/auc': train_auc, 'val/loss': val_loss_computer.avg_actual_loss, 'val/max_group_loss': max(val_loss_computer.avg_group_loss), 'val/auc': val_auc}
        wandb.log(metrics_comet, commit=True)

        patience_count += 1
        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if args.scheduler and args.model != 'bert':
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        #if epoch < args.save_step:
        #    torch.save(model,  LAST_MODEL_PATH + str(epoch) + '.pth')

        if args.save_last:
            torch.save(model,  LAST_MODEL_PATH + '.pth')

        if args.save_best:
            if args.robust or args.reweight_groups:
                #curr_val_loss = max(val_loss_computer.avg_group_loss)
                curr_val_loss = val_loss_computer.avg_actual_loss
            else:
                curr_val_loss = val_loss_computer.avg_actual_loss
            logger.write(f'Current validation loss: {curr_val_loss}\n')
            #if curr_val_loss < best_val_loss:
            if val_auc > best_val_auc:
                #best_val_loss = curr_val_loss
                best_val_auc = val_auc
                torch.save(model,  BEST_MODEL_PATH + '.pth')
                logger.write(f'Best model saved at epoch {epoch}\n')
                patience_count = 0

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                    f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        logger.write('\n')

        # Early Stopping
        if patience_count > args.patience:
            break

    # RUN TESTS
    print("Performing tests. Loading model at {}".format(BEST_MODEL_PATH))
    model = torch.load('{}.pth'.format(BEST_MODEL_PATH))
    

    # Test for Skin
    if args.dataset == 'Skin':
        test_ds_atlas_clin = CSVDataset('/datasets/abissoto/edraAtlas', '/datasets/abissoto/edraAtlas/atlas-clinical-all.csv', 'image', 'label', transform=get_transform_skin(False), add_extension='.jpg')
        test_ds_atlas_derm = CSVDataset('/datasets/abissoto/edraAtlas', '/datasets/abissoto/edraAtlas/atlas-dermato-all.csv', 'image', 'label',transform=get_transform_skin(False), add_extension='.jpg')
        test_ds_ph2 = CSVDataset('/datasets/abissoto/ph2images/', '/datasets/abissoto/ph2images/ph2.csv', 'image', 'label',transform=get_transform_skin(False), add_extension='.jpg')
        test_ds_padufes = CSVDataset('/datasets/abissoto/pad-ufes/', '/datasets/abissoto/pad-ufes/padufes-test-wocarc.csv', 'img_id', 'label',transform=get_transform_skin(False), add_extension=None)

        shuffle = False
        data_sampler = None
        num_workers = 8
        dataloaders_atlas_dermato = {
            'val': DataLoader(test_ds_atlas_derm, batch_size=4,
                                    shuffle=shuffle, num_workers=num_workers,
                                    sampler=data_sampler, pin_memory=True)
        }
        dataloaders_atlas_clin = {
            'val': DataLoader(test_ds_atlas_clin, batch_size=4,
                                    shuffle=shuffle, num_workers=num_workers,
                                    sampler=data_sampler, pin_memory=True)
        }
        dataloaders_ph2 = {
            'val': DataLoader(test_ds_ph2, batch_size=4,
                                    shuffle=shuffle, num_workers=num_workers,
                                    sampler=data_sampler, pin_memory=True)
        }
        dataloaders_padufes = {
            'val': DataLoader(test_ds_padufes, batch_size=4,
                                    shuffle=shuffle, num_workers=num_workers,
                                    sampler=data_sampler, pin_memory=True)
        }

        atlas_dermato_auc = run_eval(
                epoch, model,
                dataloaders_atlas_dermato['val'],
                args)
        atlas_clin_auc = run_eval(
                epoch, model,
                dataloaders_atlas_clin['val'],
                args)
        ph2_auc = run_eval(
                epoch, model,
                dataloaders_ph2['val'],
                args)
        padufes_auc = run_eval(
                epoch, model,
                dataloaders_padufes['val'],
                args)
        metrics_ood = {'atlas_dermato/auc': atlas_dermato_auc,  'atlas_clin/auc': atlas_clin_auc,  'ph2/auc': ph2_auc, 'padufes/auc': padufes_auc}
        wandb.log(metrics_ood, commit=True)
        
        test_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['test_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha)
        test_auc = run_epoch(
            epoch, model, optimizer,
            dataset['test_loader'],
            test_loss_computer,
            None, test_csv_logger, args,
            is_training=False) 
        metrics_test = {'test/auc': test_auc}
        wandb.log(metrics_test, commit=True)

    elif args.dataset == 'COCOonPlaces':
        pickle_dir = '/datasets/abissoto/cocoplaces'
        testiid_coco = h5py.File(pickle_dir + '/idtest.h5py', 'r')
        ood_coco = h5py.File(pickle_dir + '/oodtest.h5py', 'r')
        sg_coco = h5py.File(pickle_dir + '/sgtest.h5py', 'r')

    elif args.dataset == 'Biased':

        #for args.test_noise in ['none', 'cnoise80']:
        if args.test_noise is None or args.test_noise == 'none':
            if "tiny" in args.exp_name.lower():
                pickle_dir = '/datasets/abissoto/tinyimagenet_data/'
            elif "positionenvs" in args.exp_name.lower():
                pickle_dir = '/datasets/abissoto/imagenet-200-positionenvs/'
            elif "large" in args.exp_name.lower():
                pickle_dir = '/datasets/abissoto/imagenet-200/'
            elif "skin" in args.exp_name.lower():
                pickle_dir = '/datasets/abissoto/skin_squared/'
        elif args.test_noise == 'cnoise80':
            pickle_dir = '/datasets/abissoto/imagenet-200-noisy/'

        if args.test_noise is None:
            if args.color_noise is not None:
                exp_type = "_cnoise{}".format(args.color_noise)
            else:
                exp_type = ""

            if args.random_square_position:
                exp_type += "_position"

            if args.square_size is not None:
                exp_type += "_squared{}".format(args.square_size)
            else:
                exp_type += "_squared"

        elif args.test_noise == 'none':
            exp_type = '_squared{}'.format(args.square_size)
        elif args.test_noise == 'cnoise80':
            exp_type = '_cnoise80_position_squared{}'.format(args.square_size)
            

        
        if "tiny" in args.exp_name.lower() or "large" in args.exp_name.lower(): 
            #train_data_to_dump = pickle.load(open(os.path.join(pickle_dir, 'train' + exp_type + '_{}_{}_bf{}.pickle'.format(args.sc1, args.sc2, args.bias_factor)), 'rb'))
            test_tdd_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tdd_{}_{}.pickle'.format(args.sc1, args.sc2)), 'rb'))
            test_tdo_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tdo_{}_{}.pickle'.format(args.sc1, args.sc2)), 'rb'))
            test_tds_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tds_{}_{}.pickle'.format(args.sc1, args.sc2)), 'rb'))
            test_tdn_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tdn_{}_{}.pickle'.format(args.sc1, args.sc2)), 'rb'))
            test_tsd_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tsd_{}_{}.pickle'.format(args.sc1, args.sc2)), 'rb'))
            test_tso_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tso_{}_{}.pickle'.format(args.sc1, args.sc2)), 'rb'))
            test_tss_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tss_{}_{}.pickle'.format(args.sc1, args.sc2)), 'rb'))
            test_tsn_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tsn_{}_{}.pickle'.format(args.sc1, args.sc2)), 'rb')) 
        elif "skin" in args.exp_name.lower():
            test_tdd_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tdd.pickle'), 'rb'))
            test_tdo_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tdo.pickle'), 'rb'))
            test_tds_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tds.pickle'), 'rb'))
            test_tdn_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test_squared63_tdn.pickle'), 'rb'))
            test_tsd_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tsd.pickle'), 'rb'))
            test_tso_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tso.pickle'), 'rb'))
            test_tss_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tss.pickle'), 'rb'))
            test_tsn_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test_squared63_tsn.pickle'), 'rb'))
            test_tsd2_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tsd2.pickle'), 'rb'))
            test_tso2_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tso2.pickle'), 'rb'))
            test_tss2_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test' + exp_type + '_tss2.pickle'), 'rb'))
            test_tsn2_to_dump = pickle.load(open(os.path.join(pickle_dir, 'test_squared63_tsn2.pickle'), 'rb'))  

        tdd_ds = BiasedDataset(data=test_tdd_to_dump)
        tdo_ds = BiasedDataset(data=test_tdo_to_dump)
        tds_ds = BiasedDataset(data=test_tds_to_dump)
        tdn_ds = BiasedDataset(data=test_tdn_to_dump)
        tsd_ds = BiasedDataset(data=test_tsd_to_dump)
        tso_ds = BiasedDataset(data=test_tso_to_dump)
        tss_ds = BiasedDataset(data=test_tss_to_dump)
        tsn_ds = BiasedDataset(data=test_tsn_to_dump)

        shuffle=False
        data_sampler=None
        num_workers=8
        batch_size = 16
        tds_dl = DataLoader(tds_ds, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers,sampler=data_sampler, pin_memory=True)
        tdo_dl = DataLoader(tdo_ds, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers,sampler=data_sampler, pin_memory=True)
        tdd_dl = DataLoader(tdd_ds, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers,sampler=data_sampler, pin_memory=True)
        tdn_dl = DataLoader(tdn_ds, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers,sampler=data_sampler, pin_memory=True)
        tss_dl = DataLoader(tss_ds, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers,sampler=data_sampler, pin_memory=True)
        tso_dl = DataLoader(tso_ds, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers,sampler=data_sampler, pin_memory=True)
        tsd_dl = DataLoader(tsd_ds, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers,sampler=data_sampler, pin_memory=True)
        tsn_dl = DataLoader(tsn_ds, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers,sampler=data_sampler, pin_memory=True)
         
        tds_auc, tds_acc = run_eval(epoch, model, tds_dl, args)
        tdo_auc, tdo_acc = run_eval(epoch, model, tdo_dl, args)
        tdd_auc, tdd_acc = run_eval(epoch, model, tdd_dl, args)
        tdn_auc, tdn_acc = run_eval(epoch, model, tdn_dl, args)
        tss_auc, tss_acc = run_eval(epoch, model, tss_dl, args)
        tso_auc, tso_acc = run_eval(epoch, model, tso_dl, args)
        tsd_auc, tsd_acc = run_eval(epoch, model, tsd_dl, args)
        tsn_auc, tsn_acc = run_eval(epoch, model, tsn_dl, args)

        #metrics_ood_td = {'tds_{}/auc'.format(test_noise): tds_auc, 'tds_{}/acc'.format(test_noise): tds_acc, 'tdo_{}/auc'.format(test_noise): tdo_auc, 'tdo_{}/acc'.format(test_noise): tdo_acc, 'tdd_{}/auc'.format(test_noise): tdd_auc, 'tdd_{}/acc'.format(test_noise): tdd_acc, 'tdn_{}/auc'.format(test_noise): tdn_auc, 'tdn_{}/acc'.format(test_noise): tdn_acc}
        metrics_ood_td = {'tds/auc': tds_auc, 'tds/acc': tds_acc, 'tdo/auc': tdo_auc, 'tdo/acc': tdo_acc, 'tdd/auc': tdd_auc, 'tdd/acc': tdd_acc, 'tdn/auc': tdn_auc, 'tdn/acc': tdn_acc}
        wandb.log(metrics_ood_td, commit=True)

        #metrics_ood_ts = {'tss_{}/auc'.format(test_noise): tss_auc, 'tss_{}/acc'.format(test_noise): tss_acc, 'tso_{}/auc'.format(test_noise): tso_auc, 'tso_{}/acc'.format(test_noise): tso_acc, 'tsd_{}/auc'.format(test_noise): tsd_auc, 'tsd_{}/acc'.format(test_noise): tsd_acc, 'tsn_{}/auc'.format(test_noise): tsn_auc, 'tsn_{}/acc'.format(test_noise): tsn_acc}
        metrics_ood_ts = {'tss/auc': tss_auc, 'tss/acc': tss_acc, 'tso/auc': tso_auc, 'tso/acc': tso_acc, 'tsd/auc': tsd_auc, 'tsd/acc': tsd_acc, 'tsn/auc': tsn_auc, 'tsn/acc': tsn_acc}
        wandb.log(metrics_ood_ts, commit=True)

    if "skin" in args.exp_name.lower():
        tsd2_ds = BiasedDataset(data=test_tsd2_to_dump)
        tso2_ds = BiasedDataset(data=test_tso2_to_dump)
        tss2_ds = BiasedDataset(data=test_tss2_to_dump)
        tsn2_ds = BiasedDataset(data=test_tsn2_to_dump)
   
        tss2_dl = DataLoader(tss2_ds, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers,sampler=data_sampler, pin_memory=True)
        tso2_dl = DataLoader(tso2_ds, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers,sampler=data_sampler, pin_memory=True)
        tsd2_dl = DataLoader(tsd2_ds, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers,sampler=data_sampler, pin_memory=True)
        tsn2_dl = DataLoader(tsn2_ds, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers,sampler=data_sampler, pin_memory=True)

        tss2_auc, tss2_acc = run_eval(epoch, model, tss2_dl, args)
        tso2_auc, tso2_acc = run_eval(epoch, model, tso2_dl, args)
        tsd2_auc, tsd2_acc = run_eval(epoch, model, tsd2_dl, args)
        tsn2_auc, tsn2_acc = run_eval(epoch, model, tsn2_dl, args)

        metrics_ood_ts2 = {'tss2/auc': tss2_auc, 'tss2/acc': tss2_acc, 'tso2/auc': tso2_auc, 'tso2/acc': tso2_acc, 'tsd2/auc': tsd2_auc, 'tsd2/acc': tsd2_acc, 'tsn2/auc': tsn2_auc, 'tsn2/acc': tsn2_acc}
        wandb.log(metrics_ood_ts2, commit=True)


