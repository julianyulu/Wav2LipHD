import torch
from torch import nn
from torch.nn import functional as F

from .layers.conv import Conv2DTranspose, Conv2D

import pdb 
class Wav2LipHD(nn.Module):
    def __init__(self):
        super(Wav2LipHD, self).__init__()
        
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2D(6, 16, kernel_size=7, stride=1, padding=3, batchnorm=True)), # 192,192  n, n
            
            nn.Sequential(Conv2D(16, 32, kernel_size=3, stride=2, padding=1, batchnorm=True), # 96,96  n/2, n/2
            Conv2D(32, 32, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(32, 32, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True)),
            nn.Sequential(Conv2D(32, 64, kernel_size=3, stride=2, padding=1, batchnorm=True), # 48,48  n/2, n/2
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True)),
            nn.Sequential(Conv2D(64, 128, kernel_size=3, stride=2, padding=1, batchnorm=True),# 24,24  n/4, n/4
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True)),
            nn.Sequential(Conv2D(128, 256, kernel_size=3, stride=2, padding=1, batchnorm=True),   # 12,12  n/8, n/8 
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True)),

            nn.Sequential(Conv2D(256, 512, kernel_size=3, stride=2, padding=1, batchnorm=True),       # 6,6  n/16, n/16
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True)),

            nn.Sequential(Conv2D(512, 1024, kernel_size=3, stride=2, padding=1, batchnorm=True),     # 3,3  n/32, n/32
            Conv2D(1024, 1024, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),),
            
            nn.Sequential(Conv2D(1024, 1024, kernel_size=3, stride=1, padding=0, batchnorm=True),     # 1, 1 
            Conv2D(1024, 1024, kernel_size=1, stride=1, padding=0, batchnorm=True)),])

        self.audio_encoder = nn.Sequential(
            Conv2D(1, 32, kernel_size=3, stride=1, padding=1, batchnorm=True),
            Conv2D(32, 32, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(32, 32, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),

            Conv2D(32, 64, kernel_size=3, stride=(3, 1), padding=1, batchnorm=True),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),

            Conv2D(64, 128, kernel_size=3, stride=3, padding=1, batchnorm=True),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),

            Conv2D(128, 256, kernel_size=3, stride=(3, 2), padding=1, batchnorm=True),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),

            Conv2D(256, 512, kernel_size=3, stride=1, padding=0, batchnorm=True),
            Conv2D(512, 512, kernel_size=1, stride=1, padding=0, batchnorm=True),

            Conv2D(512, 1024, kernel_size=1, stride=1, padding=0, batchnorm=True),
            Conv2D(1024, 1024, kernel_size=1, stride=1, padding=0, batchnorm=True),)

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2D(1024, 1024, kernel_size=1, stride=1, padding=0, batchnorm=True),),

            nn.Sequential(Conv2DTranspose(2048, 1024, kernel_size=3, stride=1, padding=0, batchnorm=True), # 3,3
            Conv2D(1024, 1024, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),),

            nn.Sequential(Conv2DTranspose(2048, 1024, kernel_size=3, stride=2, padding=1, batchnorm=True, output_padding=1),
            Conv2D(1024, 1024, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(1024, 1024, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),), # 6, 6

            nn.Sequential(Conv2DTranspose(1536, 768, kernel_size=3, stride=2, padding=1, batchnorm=True, output_padding=1),
            Conv2D(768, 768, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(768, 768, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),), # 12, 12

            nn.Sequential(Conv2DTranspose(1024, 512, kernel_size=3, stride=2, padding=1, batchnorm=True, output_padding=1),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),), # 24, 24

            nn.Sequential(Conv2DTranspose(640, 320, kernel_size=3, stride=2, padding=1, batchnorm=True, output_padding=1), 
            Conv2D(320, 320, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(320, 320, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),), # 48, 48

            nn.Sequential(Conv2DTranspose(384, 192, kernel_size=3, stride=2, padding=1, batchnorm=True, output_padding=1),
            Conv2D(192, 192, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(192, 192, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),), # 96,96

            nn.Sequential(Conv2DTranspose(224, 112, kernel_size=3, stride=2, padding=1, batchnorm=True, output_padding=1),
            Conv2D(112, 112, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(112, 112, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),),]) # 192,192

        
        self.output_block = nn.Sequential(Conv2D(128, 64, kernel_size=3, stride=1, padding=1, batchnorm=True),
                                          nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid()) 

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 1024, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs

class Wav2LipHD_disc(nn.Module):
    def __init__(self):
        super(Wav2LipHD_disc, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2D(3, 16, kernel_size=7, stride=1, padding=3)), # 96, 192
            
            nn.Sequential(Conv2D(16, 32, kernel_size=5, stride=(1, 2), padding=2), # 96, 96
            Conv2D(32, 32, kernel_size =5, stride=1, padding=2)),

            nn.Sequential(Conv2D(32, 64, kernel_size=5, stride=2, padding=2), # 48,48
            Conv2D(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(Conv2D(64, 128, kernel_size=5, stride=2, padding=2),    # 24,24
            Conv2D(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(Conv2D(128, 256, kernel_size=5, stride=2, padding=2),   # 12,12
            Conv2D(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(Conv2D(256, 512, kernel_size=3, stride=2, padding=1),       # 6,6
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(Conv2D(512, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1),),
            
            nn.Sequential(Conv2D(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            Conv2D(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def forward(self, face_sequences):
        # face_sequences: [B, C, T, H, W] 
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
        return self.binary_pred(x).view(len(x), -1)

    


class Wav2LipHD_disc_patch(nn.Module):
    def __init__(self):
        super(Wav2LipHD_disc_patch, self).__init__()
        self.patches = 72
        self.face_encoder_blocks = nn.ModuleList([
            Conv2D(3, 16, kernel_size=3, stride=1, padding=1), # 96, 192
            
            Conv2D(16, 32, kernel_size=3, stride=2, padding=1), # 48, 96
            Conv2D(32, 64, kernel_size=3, stride=2, padding=1), # 24, 48
            Conv2D(64, 128, kernel_size=3, stride=2, padding=1), # 12, 24
            Conv2D(128, 256, kernel_size=3, stride=2, padding=1), # 6, 12
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),# 6, 12
            nn.Sigmoid()
            ]) # receptive field: (65, 65) 
            
    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def forward(self, face_sequences):
        # face_sequences: [B, C, T, H, W] 
        face_sequences = self.to_2d(face_sequences) # (B*T, C, H, W)
        face_sequences = self.get_lower_half(face_sequences)
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
        # x: [B*T, 1, 6, 12]
        x = torch.squeeze(x, 1) # [B *T, 6, 12]
        x = torch.cat([x[:, :, i] for i in range(x.size(2))], dim = 1) # [B*T, 72]
        x = torch.flatten(x)
        return x 

