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
from .loss import compute_loss_with_l1, compute_loss_with_l2
from .plot import plot_history
from .model import adjust_learning_rate

def get_errors(y_true, y_pred): # return rmse, mae, maxae
    return [np.sqrt(mse(y_true, y_pred)),
            mae(y_true, y_pred),
            np.max(np.abs(y_true - y_pred))]


def train_single_epoch_model(model, criterion,
                             dataloader, optimizer,
                             use_gpu=torch.cuda.is_available()):
    for inputs, ys in tqdm(dataloader, desc='Training model',
                           unit_scale=dataloader.batch_size):
        if use_gpu:
            inputs = [tsr.cuda() for tsr in inputs]
            ys = ys.cuda()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = compute_loss_with_l1(criterion(outputs, ys), model)
#         loss = compute_loss_with_l2(criterion(outputs, ys), model)
        loss.backward()
        optimizer.step()

def test_model(model, dataloader, use_gpu=torch.cuda.is_available(),
               return_y=False, freeze_mask=True, new_p=None):
    
    y_true = []
    y_pred = []
    if freeze_mask:
        model.freeze_mask_dropout(new_p)
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
    if return_y:
        return np.concatenate(y_true), np.concatenate(y_pred)
    return get_errors(np.concatenate(y_true), np.concatenate(y_pred))


def train_model(model, criterion, dataloaders, optimizer,
                n_epochs, check_every, patience_max,
                use_gpu=torch.cuda.is_available(), print_const=''):
    if n_epochs % check_every:
        n_epochs += check_every - n_epochs % check_every
    current_error = 1e+10
    patience = 0
    logs = {'train': list(), 'val': list(), 'check_every': check_every}
    started_at = time()
    try:
        for epoch in range(n_epochs):
            ##############################################################################
            #                           STATISTICS                                       #
            ##############################################################################
            lr = adjust_learning_rate(optimizer, epoch+1) 
            clear_output(True)
            print(print_const, flush=True)
            print('Train Epoch #{} out of {}.'.format(epoch + 1, n_epochs), flush=True)
            print('Learning rate: {}'.format(lr), flush=True)
            if epoch > check_every:         
                print('Last train rmse: {:.2f}'.format(logs['train'][-1][0]), flush=True)
                print('Last val rmse:   {:.2f}'.format(logs['val'][-1][0]), flush=True)
            ##############################################################################
            #                             TRAINING                                       #
            ##############################################################################
            train_single_epoch_model(model, criterion, dataloaders['TrainPool'], optimizer, use_gpu)
            ##############################################################################
            #                              ERRORS                                        #
            ##############################################################################
            if ((epoch + 1) % check_every == 0):
                train_errors = test_model(model, dataloaders['TrainPool'], use_gpu, new_p=0)
                logs['train'] += [train_errors]
                val_errors = test_model(model, dataloaders['Val'], use_gpu, new_p=0)
                logs['val'] += [val_errors]
                logs['epochs'] = epoch + 1
                if train_errors[0] > current_error:
                    patience += 1
                if patience > patience_max:
                    print('Early stopping in epoch {}'.format(epoch + 1))
                    break
                current_error = train_errors[0]
            ##############################################################################
            #                   END ONE EPOCH  --- REPEAT                                #
            ##############################################################################
    except KeyboardInterrupt:
        print('Interrupted!')
        epoch = 0
#     clear_output(True)
    total_time = time() - started_at
    logs['time'] = total_time
    logs['epochs'] = epoch + 1
    print('-' * 20)
    print('Training complete in {:.0f} mins {:.0f} seconds with {} epochs'.format(total_time // 60,
                                                                                  total_time % 60,
                                                                                  epoch+1), flush=True)
    
    try:
        ##############################################################################
        #                                TEST                                        #
        ##############################################################################
        test_errors = test_model(model, dataloaders['Test'], use_gpu, new_p=0)
        logs['test'] = [test_errors]
    except KeyboardInterrupt:
        print('Interrupted!')
    return model, logs


def active_train_model(model, criterion, dataloaders, optimizer, acquisition_fun, 
                       n_al_epochs, n_train_epochs, n_val, use_gpu=torch.cuda.is_available()):
    dataloaders['TrainPool'].set_mode('pool')
    new_point_size = len(dataloaders['TrainPool'].dataset.pool_idx) // (2 * n_al_epochs)
    print('Set size of new points: {}'.format(new_point_size))
    logs = {'train': list(), 'val': list()}
    train_logs = []
    started_at = time()
    train_params = {
        'criterion': criterion,
        'dataloaders': dataloaders,
        'optimizer': optimizer,
        'n_epochs': n_train_epochs,
        'check_every': 100,
        'patience_max': 3
    }
    try:
        for epoch in range(n_al_epochs):
            ##############################################################################
            #                           STATISTICS                                       #
            ##############################################################################
            clear_output(True)
            print_const = 'AL Epoch #{} out of {}.\n'.format(epoch + 1, n_al_epochs)
            if epoch > 0:
                print_const += 'Last al train rmse: {:.2f}\n'.format(logs['train'][-1][0][0])
                print_const += 'Last val rmse:   {:.2f}\n'.format(logs['val'][-1][0][0])
            ##############################################################################
            #                  CHOOSE NEW POINTS FROM POOL                               #
            ##############################################################################
            dataloaders['TrainPool'].set_mode('pool')
            new_idx = acquisition_fun(new_point_size, model, dataloaders['TrainPool'], use_gpu)
            dataloaders['TrainPool'].update_indexes(new_idx)
            ##############################################################################
            #                             TRAINING                                       #
            ##############################################################################
            dataloaders['TrainPool'].set_mode('train')
            train_params['print_const'] = print_const
            model, train_log = train_model(model=model, **train_params)
            train_logs.append(train_log)
            ##############################################################################
            #                               ERRORS                                       #
            ##############################################################################
            train_errors = []
            val_errors = []
            for i in range(n_val):
                train_errors.append(test_model(model, dataloaders['TrainPool'], use_gpu, new_p=0))
                val_errors.append(test_model(model, dataloaders['Val'], use_gpu, new_p=0))
            logs['train'] += [train_errors]
            logs['val'] += [val_errors]
            logs['epochs'] = epoch + 1
            ##############################################################################
            #                   END ONE EPOCH  --- REPEAT                                #
            ##############################################################################
    except KeyboardInterrupt:
        print('Interrupted!')
        epoch = 0
    
#     clear_output(True)
    total_time = time() - started_at
    print('-' * 20)
    print('Training complete in {:.0f} mins {:.0f} seconds with {} epoch'.format(total_time // 60,
                                                                                 total_time % 60,
                                                                                 epoch+1), flush=True)
    
    try:
        ##############################################################################
        #                                TEST                                        #
        ##############################################################################
        test_errors = test_model(model, dataloaders['Test'], use_gpu, new_p=0)
        logs['test'] = [test_errors]
    except KeyboardInterrupt:
        print('Interrupted!')
    return model, logs, train_logs
