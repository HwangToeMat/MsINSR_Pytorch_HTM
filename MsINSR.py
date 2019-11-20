import torch
import torch.nn as nn
import torch.functional as F

class INRB(nn.Module):
    def __init__(self, in_channels):
        super(INRB, self).__init__()
        self.branch_0 = nn.Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False)
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            nn.Conv2d(32, 48, 3, stride=1, padding=1, bias=False),
            nn.Conv2d(48, 64, 3, stride=1, padding=1, bias=False)
        )
        self.residual = nn.Conv2d(128, in_channels, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        out_res = self.residual(torch.cat([branch_0, branch_1, branch_2], 1))
        out_res += x
        output = self.relu(out_res)
        return output

class MsINSR(nn.Module):
    def __init__(self):
        super(MsINSR, self).__init__()
        self.input = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.INRB1 = INRB(64)
        self.INRB2 = INRB(64)
        self.INRB3 = INRB(64)
        self.INRB4 = INRB(64)
        self.INRB5 = INRB(64)
        self.output = nn.Sequential(
            nn.Conv2d(64*(5+1), 64, 1, stride=1, padding=0),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        R_0 = self.input(x)
        R_1 = self.INRB1(R_0)
        R_2 = self.INRB2(R_1)
        R_3 = self.INRB3(R_2)
        R_4 = self.INRB4(R_3)
        R_5 = self.INRB5(R_4)
        output = self.output(torch.cat([R_0, R_1, R_2, R_3, R_4, R_5], 1))
        output = (output+1)/2
        return output
