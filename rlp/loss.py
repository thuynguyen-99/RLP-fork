import torch
import torch.nn as nn
from pytorch_msssim import ssim
import torch.nn.functional as F

class Loss(nn.Module):
    """Charbonnier + SSIM + Edge"""
    def __init__(self, w_l1=1.0, w_ssim=0.2, w_edge=0.1, eps=1e-3):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_edge = w_edge
        self.eps = eps

        gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("gx", gx)
        self.register_buffer("gy", gy)

    def charbonnier(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

    def edge_loss(self, x, y):
        if x.size(1) > 1:
            x = 0.2989*x[:,0:1] + 0.5870*x[:,1:2] + 0.1140*x[:,2:3]
            y = 0.2989*y[:,0:1] + 0.5870*y[:,1:2] + 0.1140*y[:,2:3]
        gx1 = F.conv2d(x, self.gx, padding=1)
        gy1 = F.conv2d(x, self.gy, padding=1)
        gx2 = F.conv2d(y, self.gx, padding=1)
        gy2 = F.conv2d(y, self.gy, padding=1)
        grad1 = torch.sqrt(gx1**2 + gy1**2 + 1e-6)
        grad2 = torch.sqrt(gx2**2 + gy2**2 + 1e-6)
        return torch.mean(torch.abs(grad1 - grad2))

    def forward(self, pred, target):
        l1 = self.charbonnier(pred, target)
        l_ssim = 1 - ssim(pred, target, data_range=1.0, size_average=True)
        l_edge = self.edge_loss(pred, target)
        loss = self.w_l1*l1 + self.w_ssim*l_ssim + self.w_edge*l_edge
        return loss
