import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


def freeze_module(model):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0       
    
def save_module(module, filepath):
    torch.save(module.state_dict(), filepath)

def load_module(module, filepath, freeze=True):
    module.load_state_dict(torch.load(filepath))
    if freeze:
        freeze_module(module)
    return module

class ConstDropout(nn.Dropout):
        def __init__(self, num_features, p=0.5, inplace=False):
            super(ConstDropout, self).__init__(p, inplace)
            self.mask = torch.ones(num_features)
            self.num_features = num_features

        def forward(self, input):
            return F.dropout(input, self.p, self.training, self.inplace) * torch.stack([self.mask]*input.size(0))
        
        def set_const_mask(self):
            self.mask = torch.distributions.Bernoulli(torch.full((self.num_features,), self.p)).sample()
        
        def unset_const_mask(self):
            self.mask = torch.ones(self.num_features)
            
            
class ALModel(nn.Module):
    def __init__(self, layers, p=0.05, nonlinearity=None):
        super(ALModel, self).__init__()
        
        if nonlinearity is None:
            nonlinearity = nn.LeakyReLU(negative_slope=0.2)
        
        nn_laeyrs = []
        for view_l, view_r in zip(layers, layers[1:]):
            nn_laeyrs.append(nn.Linear(view_l, view_r))
#             nn_laeyrs.append(nn.BatchNorm1d(view_r))
            nn_laeyrs.append(nonlinearity)
            nn_laeyrs.append(ConstDropout(view_r, p=p))
#             nn_laeyrs.append(nn.Dropout(p=p))
        
        nn_laeyrs.append(nn.Linear(layers[-1], 1))
        nn_laeyrs.append(nonlinearity)
        self.net = nn.Sequential(*nn_laeyrs)

    def forward(self, input):
        output = self.net(input)
        return output

    def freeze_mask_dropout(self):
        for m in self.net.modules():
            if isinstance(m, nn.Dropout):
                m.set_const_mask()
                
    def unfreeze_mask_dropout(self):
        for m in self.net.modules():
            if isinstance(m, nn.Dropout):
                m.unset_const_mask()
    