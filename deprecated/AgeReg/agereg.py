import os
from os import path

from IPython.display import clear_output
from tqdm import tqdm
from time import time

import numpy as np
from sklearn.model_selection import train_test_split



import torch
from torch import nn
from torch.functional import F


from torchvision.models import vgg
from torchvision.datasets.folder import default_loader as image_loader
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader




def save_module(module, filepath):
    torch.save(module.state_dict(), filepath)


def load_module(module, filepath, freeze=True):
    module.load_state_dict(torch.load(filepath))
    return module


##################################################################################
#                                   DATA                                         #
##################################################################################

class ImageAgeDataset(Dataset):
    
    def __init__(self, files, transform, get_age):
        super(ImageAgeDataset, self).__init__()
        self.files = files
        self.transform = transform
        self.get_age = get_age
        self.id_to_file = dict(enumerate(self.files))
        self.file_to_id = {v:k for k, v in self.id_to_file.items()}
        self._length = len(self.id_to_file)
        
    def _load_image(self, index):
        img = image_loader(self.id_to_file[index])
        img_tensor = self.transform(img)
        return img_tensor
                            
    def _load_age(self, index):
        return torch.Tensor([self.get_age(self.id_to_file[index])])
    
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self._load_image(index), self._load_age(index)


def pad_to_square(image):
    h, w = image.size
    padding = abs(h - w)
    if h < w:
        image = transforms.Pad((padding // 2, 0), (255,) * 3)(image)
    elif w < h:
        image = transforms.Pad((0, padding // 2), (255,) * 3)(image)
    return image

def split_array(array, splits, seed=42):
    """ Randomly split one Container into several ones with specified proportions. """
    assert sum(splits) == 1
    arrs = [array]
    for i in range(len(splits) - 1):
        arrs = arrs[:i] + train_test_split(arrs[i],
                                           train_size=splits[i] / sum(splits[i:]),
                                           random_state=seed)
    return arrs

##################################################################################
#                              AUGMENTATION                                      #
##################################################################################
def transforms_black_box(img):
    _, h, v = img.size()
    mode = np.random.choice([1,2,3,4])
    mask = torch.ones((3, h, v))
    if mode == 1:
        mask[:, h // 8 : h // 2, v // 8 : v // 2] = 0
    if mode == 2:
        mask[:, h // 8 : h // 2, v // 2 : 7*v // 8] = 0
    if mode == 3:
        mask[:, h // 2 : 7*h // 8, v // 8 : v // 2] = 0
    if mode == 4:
        mask[:, h // 2 : 7*h // 8, v // 2 : 7*v // 8] = 0
    return img * mask

def get4box(imgs):
    b, c, h, v = imgs.size()
    mask1 = torch.ones((b, c, h, v)).cuda()
    mask1[:, :, h // 8 : h // 2, v // 8 : v // 2] = 0
    mask2 = torch.ones((b, c, h, v)).cuda()
    mask2[:, :, h // 8 : h // 2, v // 2 : 7*v // 8] = 0
    mask3 = torch.ones((b, c, h, v)).cuda()
    mask3[:, :, h // 2 : 7*h // 8, v // 8 : v // 2] = 0
    mask4 = torch.ones((b, c, h, v)).cuda()
    mask4[:, :, h // 2 : 7*h // 8, v // 2 : 7*v // 8] = 0
    return torch.cat([imgs * mask1, imgs * mask2, imgs * mask3, imgs * mask4])

##################################################################################
#                                   LOSS                                         #
##################################################################################
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
##################################################################################
#                             REGULARIZATION                                     #
##################################################################################
def compute_loss_with_l1(loss, model, coef=2e-5):
    l1_loss = sum(F.l1_loss(p, torch.zeros_like(p))
                  for p in model.parameters() if p.requires_grad)
    return loss + coef * l1_loss

def compute_loss_with_l2(loss, model, coef=2e-5):
    l2_loss = sum(torch.norm(p) for p in model.parameters() if p.requires_grad)
    return loss + coef * l2_loss
##################################################################################
#                                  MODEL                                         #
##################################################################################
class ConstDropout(nn.Dropout):
    
    def __init__(self, num_features, p=0.5, inplace=False):
        super(ConstDropout, self).__init__(p, inplace)
        self.mask = torch.ones(num_features)
        self.freezed = False
        self.num_features = num_features

    def forward(self, input):
        if not self.freezed:
            return F.dropout(input, self.p, self.training, self.inplace)
        else:
            return input * torch.stack([self.mask]*input.size(0)).type(input.type())

    def freeze_mask(self, new_p=None):
        self.freezed = True
        new_p = new_p or self.p
        keep_p = 1 - new_p
        self.mask = torch.distributions.Bernoulli(torch.full((self.num_features,), keep_p)).sample()/keep_p

    def unfreeze_mask(self):
        self.freezed = False
        self.mask = torch.ones(self.num_features)
        
class ModelWithFreeze(nn.Module):
    
    def freeze_mask_dropout(self, new_p=None):
        for m in self.modules():
            if isinstance(m, ConstDropout):
                m.freeze_mask(new_p)
                
    def unfreeze_mask_dropout(self):
        for m in self.modules():
            if isinstance(m, ConstDropout):
                m.unfreeze_mask()

                
class AgeRegModel(ModelWithFreeze):
    
    def __init__(self, features, regression, freeze_features=True, border=None):
        super(AgeRegModel, self).__init__()
        
        self.features = features
        self.regression = regression
        
        if freeze_features:
            self._freezee_weights(self.features, border)
            
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regression(x)
        return x
    
    def _freezee_weights(self, layer, border):
        if border is None:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            print('Non Freeze {} leyer and deeper'.format(border))
            for l in layer.named_modules():
                if l[0] > border:
                    for param in l[1].parameters():
                        param.requires_grad = False
        for module in layer.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0   
        
        
def make_layers_old(num_features, p=0.05,
                negative_slope=0.05,
                nonlinearity=None,
                regression=True):
    
    assert len(num_features) >= 2, 'Net\'s deep should be longer'
    
    if nonlinearity is None:
        nonlinearity = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        
    laeyrs = []
    for view_l, view_r in zip(num_features[:-2], num_features[1:]):
        laeyrs.append(nn.Linear(view_l, view_r))
        laeyrs.append(nonlinearity)
        laeyrs.append(ConstDropout(view_r, p=p))
    
    laeyrs.append(nn.Linear(num_features[-2], num_features[-1]))
    laeyrs.append(nonlinearity)
    
    if regression:
        laeyrs.append(nn.Linear(num_features[-1], 1))
    
    return nn.Sequential(*laeyrs)




def adjust_learning_rate(optimizer, c_epoch, n_epochs, lr_init, lr_decay_epoch=[30], lr_step=0.1):
    # sets the learning rate to the initial LR decayed
    if sum(c_epoch == epoch for epoch in lr_decay_epoch) > 0: 
        state_dict = optimizer.state_dict()
        for param_group in state_dict['param_groups']:
            lr = param_group['lr'] * lr_step
            param_group['lr'] = lr
        optimizer.load_state_dict(state_dict)
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    return lr