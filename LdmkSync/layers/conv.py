import torch
from torch import nn

class Conv2D(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual = False, batchnorm = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if batchnorm: 
            self.conv_block = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size, stride, padding, bias = True),
                nn.BatchNorm2d(cout))
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size, stride, padding))
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class Conv1D(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm1d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)
