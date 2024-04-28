import torch.nn as nn
import torch.nn.functional as F
import torch 
from torch.autograd import Variable
import torch.autograd as autograd

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class Soft_Sort(nn.Module):
    def __init__(self,tau = 1.0,pow = 1.0):
        super().__init__()
        self.tau = tau
        self.pow = pow
    def forward(self,scores):
        scores = scores.unsqueeze(-1)
        sorted = torch.sort(scores, descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().pow(self.pow).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)
        return P_hat



