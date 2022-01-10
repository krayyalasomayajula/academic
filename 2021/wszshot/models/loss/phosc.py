import torch
import torch.nn as nn
import pdb

class PhosLoss(nn.Module):
    def __init__(self,
                 loss_config):
        super(PhosLoss, self).__init__()
        self.loss = nn.MSELoss().cuda()
        
    def forward(self, pred, target):
        mse = self.loss(pred, target)
        return mse
        

class PhocLoss(nn.Module):
    def __init__(self,
                 loss_config):
        super(PhocLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss().cuda()
        
    def forward(self, pred, target):
        ce_loss = self.loss(pred, target)
        return ce_loss
        
