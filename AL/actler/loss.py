import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

def compute_loss_with_l1(loss, model, l1_coef=0.05):
    l1_loss = sum(F.l1_loss(p, torch.zeros_like(p))
                  for p in model.parameters() if p.requires_grad)
    return loss + l1_coef * l1_loss

def compute_loss_with_l2(loss, model, l2_coef=0.05):
    l2_loss = 0
    for p in model.parameters():
        if p.requires_grad:
            l2_loss += (p**2).sum()
    return loss + l2_coef * l2_loss
