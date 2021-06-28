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

class Conv2DTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding, bias = True),
            nn.BatchNorm2d(cout))
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
