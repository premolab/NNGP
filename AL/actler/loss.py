import torch
from torch import nn
import torch.nn.functional as F

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

def compute_loss_with_l1(loss, model, l1_coef=0.05):
    absolute = sum(F.l1_loss(p, torch.zeros_like(p))
                   for p in model.parameters() if p.requires_grad)
    return loss + l1_coef * absolute