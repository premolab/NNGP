import numpy as np
from scipy.optimize import rosen
from sklearn.model_selection import train_test_split

def generate_rosen_data(size, d, meshgrid=False):
    if not meshgrid:
        X = np.random.rand(size, d)
        y = rosen(X.T)[:, None]
        return X, y
    else:
        Xi, Xj = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
        X = np.vstack([Xi.ravel(), Xj.ravel()]).T
        y = rosen(X.T)
        Y = y.reshape(Xi.shape)
        return X, y[:, None], Xi, Xj, Y

def delete_subarr(arr, subarr):
    return np.array([i for i in arr if i not in subarr])

def split_array(array, splits, seed=42):
    """ Randomly split one Container into several ones with specified proportions. """
    arrs = [array]
    for i in range(len(splits) - 1):
        arrs += train_test_split(arrs.pop(),
                                 test_size=1 - splits[i]/sum(splits[i:]),
                                 random_state=seed)
    return arrs


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

#For Test and Validate
class SampleDataset(Dataset): 
    def __init__(self, X, y):
        self._length = len(X)
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)
        
    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class TrainPoolDataset(Dataset):
    def __init__(self, X=None, y=None, split=None,
                 X_train=None, y_train=None, X_pool=None, y_pool=None):
        if all([val is not None for val in (X, y, split)]):
            self.train_idx, self.pool_idx = train_test_split(np.arange(len(X)), test_size=split)
        elif all([val is not None for val in (X_train, y_train, X_pool, y_pool)]):
            self.train_idx = np.arange(0, len(X_train))
            self.pool_idx = np.arange(len(X_train), len(X_train) + len(X_pool))
            X = np.hstack([X_train, X_pool])
            y = np.hstack([y_train, y_pool])
        else:
            assert False, 'Check arguments'
        self.mode = 'train'
        self.indexes = self.train_idx
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)
    
    def set_mode(self, mode):
        self.mode = mode
        self.indexes = getattr(self, self.mode + '_idx')
    
    def update_indexes(self, idx):
        chosen_idx = self.indexes[idx]
        self.train_idx = np.concatenate([self.train_idx, chosen_idx])
        self.pool_idx = delete_subarr(self.pool_idx, chosen_idx)
    
    def __len__(self):
        self._length = len(self.indexes)
        return self._length

    def __getitem__(self, idx):
        return self.X[self.indexes[idx]], self.y[self.indexes][idx]

    
class CustomDataLoader(DataLoader):
    def set_mode(self, mode):
        self.dataset.set_mode(mode)
    
    def update_indexes(self, idx):
        self.dataset.update_indexes(idx)
    
def make_dataloaders(data, y, splits = [0.1, 0.8, 0.1, 0.1], BS=50):
    all_index = np.arange(len(data))
    train_pool_idx, test_idx, val_idx = split_array(all_index, [sum(splits[:2])] + splits[2:])
    train_pool_dataset = TrainPoolDataset(data[train_pool_idx],
                                          y[train_pool_idx],
                                          splits[1]/sum(splits[:2]))
    test_dataset = SampleDataset(data[test_idx], y[test_idx])
    val_dataset = SampleDataset(data[val_idx], y[val_idx])
    dataloaders = {'TrainPool': CustomDataLoader(train_pool_dataset, batch_size=BS),
                   'Test': DataLoader(test_dataset, batch_size=BS),
                   'Val': DataLoader(val_dataset, batch_size=BS)}
    return dataloaders

