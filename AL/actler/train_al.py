import os
import numpy as np
from tqdm import tqdm
from time import time
from IPython.display import clear_output

import torch
from torch import nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from .loss import compute_loss_with_l1

def get_errors(y_true, y_pred): # return rmse, mae, maxae
    return [np.sqrt(mse(y_true, y_pred)),
            mae(y_true, y_pred),
            np.max(np.abs(y_true - y_pred))]


def train_single_epoch_model(model, criterion, dataloader, optimizer,
                             use_gpu=torch.cuda.is_available()):
    y_true = []
    y_pred = []
    for inputs, ys in tqdm(dataloader, desc='Training model',
                           unit_scale=dataloader.batch_size):
        if use_gpu:
            inputs = [tsr.cuda() for tsr in inputs]
            ys = ys.cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = compute_loss_with_l1(criterion(outputs, ys), model)
        loss.backward()
        optimizer.step()
     
        y_true.append(ys.tolist())
        y_pred.append(outputs.tolist())
        
    return get_errors(np.concatenate(y_true), np.concatenate(y_pred))
    

def test_model(model, dataloader, use_gpu=torch.cuda.is_available(),
               freeze_mask=True, return_pred=False):
    
    y_true = []
    y_pred = []
    if freeze_mask:
        model.freeze_mask_dropout()
    with torch.autograd.no_grad():
        for inputs, ys in tqdm(dataloader, desc='Testing model',
                                   unit_scale=dataloader.batch_size):
            if use_gpu:
                inputs = [tsr.cuda() for tsr in inputs]
                ys = ys.cuda()

            outputs = model(inputs)
            y_true.append(ys.tolist())
            y_pred.append(outputs.tolist())
    if freeze_mask:
        model.unfreeze_mask_dropout()
    if return_pred:
        return np.concatenate(y_pred)
    return get_errors(np.concatenate(y_true), np.concatenate(y_pred))


def train_model(model, criterion, dataloaders, optimizer,
                n_epochs, use_gpu=torch.cuda.is_available(), freeze_mask=False):
    logs = {'train': list(), 'val': list()}
    started_at = time()
    try:
        for epoch in range(n_epochs):
            ##############################################################################
            #                           STATISTICS                                       #
            ##############################################################################
            clear_output(True)
            print('Train Epoch #{} out of {}.'.format(epoch + 1, n_epochs), flush=True)
            if epoch > 0:
                print('Last train rmse: {:.2f}'.format(logs['train'][-1][0]), flush=True)
                print('Last val rmse:   {:.2f}'.format(logs['val'][-1][0]), flush=True)
            ##############################################################################
            #                             TRAINING                                       #
            ##############################################################################
            train_errors = train_single_epoch_model(model, criterion,
                                                    dataloaders['TrainPool'],
                                                    optimizer, use_gpu)
            logs['train'] += [train_errors]
            ##############################################################################
            #                            VALIDATION                                      #
            ##############################################################################
            val_errors = test_model(model, dataloaders['Val'], use_gpu, freeze_mask)
            logs['val'] += [val_errors]
            ##############################################################################
            #                   END ONE EPOCH  --- REPEAT                                #
            ##############################################################################
    except KeyboardInterrupt:
        print('Interrupted!')
        epoch = 0
    
    clear_output(True)
    total_time = time() - started_at
    print('-' * 20)
    print('Training complete in {:.0f} mins {:.0f} seconds with {} epoch'.format(total_time // 60,
                                                                                 total_time % 60,
                                                                                 epoch+1), flush=True)
    
    try:
        ##############################################################################
        #                                TEST                                        #
        ##############################################################################
        test_errors = test_model(model, dataloaders['Test'], use_gpu, freeze_mask)
        logs['test'] = [test_errors]
    except KeyboardInterrupt:
        print('Interrupted!')
    return model, logs


def active_train_model(model, criterion, dataloaders, optimizer, acquisition_fun, new_point_size,
                       n_epochs, n_val, use_gpu=torch.cuda.is_available()):
    logs = {'train': list(), 'val': list()}
    started_at = time()
    try:
        for epoch in range(n_epochs):
            ##############################################################################
            #                           STATISTICS                                       #
            ##############################################################################
            clear_output(True)
            print('Train Epoch #{} out of {}.'.format(epoch + 1, n_epochs), flush=True)
            if epoch > 0:
                print('Last train rmse: {:.2f}'.format(logs['train'][-1][0]), flush=True)
                print('Last val rmse:   {:.2f}'.format(logs['val'][-1][0][0]), flush=True)
            ##############################################################################
            #                             TRAINING                                       #
            ##############################################################################
            dataloaders['TrainPool'].set_mode('train')
            train_errors = train_single_epoch_model(model, criterion,
                                                    dataloaders['TrainPool'],
                                                    optimizer, use_gpu)
            logs['train'] += [train_errors]
            ##############################################################################
            #                            VALIDATION                                      #
            ##############################################################################
            val_errors = []
            for i in range(n_val):
                val_errors.append(test_model(model, dataloaders['Val'], use_gpu))
            logs['val'] += [val_errors]
            ##############################################################################
            #                  CHOOSE NEW POINTS FROM POOL                               #
            ##############################################################################
            dataloaders['TrainPool'].set_mode('pool')
            new_idx = acquisition_fun(new_point_size, model, dataloaders['TrainPool'], use_gpu)
            dataloaders['TrainPool'].update_indexes(new_idx)
    except KeyboardInterrupt:
        print('Interrupted!')
        epoch = 0
    
    clear_output(True)
    total_time = time() - started_at
    print('-' * 20)
    print('Training complete in {:.0f} mins {:.0f} seconds with {} epoch'.format(total_time // 60,
                                                                                 total_time % 60,
                                                                                 epoch+1), flush=True)
    
    try:
        ##############################################################################
        #                                TEST                                        #
        ##############################################################################
        test_errors = test_model(model, dataloaders['Test'], use_gpu)
        logs['test'] = [test_errors]
    except KeyboardInterrupt:
        print('Interrupted!')
    return model, logs
