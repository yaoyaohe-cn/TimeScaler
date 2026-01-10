import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuredLoss(nn.Module):
    def __init__(self, loss_name='smoothL1', alpha=1.0, beta=1.0, gamma=1.0):
        super(StructuredLoss, self).__init__()
        
        if loss_name == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_name == 'smoothL1':
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.SmoothL1Loss() 
            
        self.alpha = alpha # Weight for Reconstruction loss
        self.beta = beta   # Weight for Trend (Approx) loss
        self.gamma = gamma # Weight for Detail loss

    def forward(self, pred, true, pred_yl, pred_yh_list, odb_module, revin_module):
        B, L, C = true.shape
        
        # 1. Normalize Ground Truth to align with model's latent space
        true_norm = revin_module(true, 'norm')  
        true_norm = true_norm.permute(0, 2, 1).reshape(B * C, 1, L)
        
        # 2. Decompose Ground Truth using OD Block
        true_yl, true_yh_list = odb_module(true_norm)
        
        # 3. Component Supervision
        loss_trend = self.criterion(pred_yl, true_yl)
        
        loss_detail = 0
        for p_h, t_h in zip(pred_yh_list, true_yh_list):
            loss_detail += self.criterion(p_h, t_h)
            
        # 4. Reconstruction Loss
        loss_recon = self.criterion(pred, true)
        
        # 5. Weighted Sum
        total_loss = (self.alpha * loss_recon) + (self.beta * loss_trend) + (self.gamma * loss_detail)
        return total_loss
