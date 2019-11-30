import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.backends.cudnn as cudnn
import numpy as np

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight = 1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        vgg_loss = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in vgg_loss.parameters():
            param.requires_grad = False
        self.vgg_loss = vgg_loss
        self.L1_loss = nn.L1Loss()
        self.BCE_stable = nn.BCEWithLogitsLoss()
        self.TV_Loss = TVLoss()

    def forward(self, SR, HR, L1_parm = 1, Percept_parm = 0, Gen_parm =0, TV_parm = 2e-8):
        # L1 Loss
        L1_loss = self.L1_loss(SR, HR)
        # VGG Loss
        VGG_loss = self.L1_loss(self.vgg_loss(SR), self.vgg_loss(HR))
        # TV Loss
        TV_loss = self.TV_Loss(SR)
        return L1_parm*L1_loss + Percept_parm*VGG_loss + TV_parm*TV_loss

def PSNR(HR, fake_img):

    mse = nn.MSELoss()
    loss = mse(HR, fake_img)
    psnr = 10 * np.log10(1 / (loss.item() + 1e-10))
    return psnr
