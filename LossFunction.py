import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.backends.cudnn as cudnn

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss

class GeneratorLoss(nn.Module):
    def __init__(self, L1_parm = 1, Percept_parm = 6e-3, Gen_parm = 1e-3):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        vgg_loss = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in vgg_loss.parameters():
            param.requires_grad = False
        self.vgg_loss = vgg_loss
        self.CharbonnierLoss = CharbonnierLoss()
        self.BCE_stable = nn.BCEWithLogitsLoss()

    def forward(self, fake_rate, real_rate, SR, HR):
        # L1 Loss
        L1_loss = self.CharbonnierLoss(SR, HR)
        # VGG Loss
        VGG_loss = self.CharbonnierLoss(self.vgg_loss(SR), self.vgg_loss(HR))
        # Relative Generator loss
        Rel_G_loss = (self.BCE_stable(real_rate - torch.mean(fake_rate), torch.zeros(fake_rate.size(0)).cuda()) +
                    self.BCE_stable(fake_rate - torch.mean(real_rate), torch.ones(fake_rate.size(0)).cuda()))/2
        return L1_parm*L1_loss + Percept_parm*VGG_loss + Gen_parm*Rel_G_loss

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.BCE_stable = nn.BCEWithLogitsLoss()

    def forward(self, fake_rate, real_rate):
        # Relative Discriminator loss
        Rel_D_loss = (self.BCE_stable(real_rate - torch.mean(fake_rate), torch.ones(fake_rate.size(0)).cuda()) +
                    self.BCE_stable(fake_rate - torch.mean(real_rate), torch.zeros(fake_rate.size(0)).cuda()))/2
        return Rel_D_loss
