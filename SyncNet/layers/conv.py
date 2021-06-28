import torch
from torch import nn

class Conv2D(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace = True))

    def forward(self, x):
        out = self.conv_block(x)
        return out 
                        
        

class Conv3D(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace = True))

    def forward(self, x):
        out = self.conv_block(x)
        return out 
    
