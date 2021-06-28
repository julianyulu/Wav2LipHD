import torch
from torch import nn
from torch.nn import functional as F

from .layers.conv import Conv2D

class SyncNetColorHD(nn.Module):
    def __init__(self):
        super(SyncNetColorHD, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2D(15, 32, kernel_size=(7, 7), stride=1, padding=3, batchnorm=True),
            # 96, 192

            Conv2D(32, 64, kernel_size=5, stride=(1, 2), padding=1, batchnorm=True),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            # 96, 96

            Conv2D(64, 128, kernel_size=3, stride=2, padding=1, batchnorm=True),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            # 48, 48

            Conv2D(128, 256, kernel_size=3, stride=2, padding=1, batchnorm=True),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            # 24, 24

            Conv2D(256, 512, kernel_size=3, stride=2, padding=1, batchnorm=True),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            # 12, 12
            
            Conv2D(512, 1024, kernel_size=3, stride=2, padding=1, batchnorm=True),
            Conv2D(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            Conv2D(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            # 6, 6

            Conv2D(1024, 1024, kernel_size=3, stride=2, padding=1, batchnorm=True),
            # 3, 3
            Conv2D(1024, 1024, kernel_size=3, stride=1, padding=0, batchnorm=True),
            # 1, 1
            Conv2D(1024, 512, kernel_size=1, stride=1, padding=0, batchnorm=True),) # 1, 1
            # note here should be 1024, 1024, according to the default style,
            # I keep it 512 so that no need to change audo branch, such that
            # both the video and audio branch yieds 512 len vectors
            
        self.audio_encoder = nn.Sequential(
            Conv2D(1, 32, kernel_size=3, stride=1, padding=1, batchnorm=True),
            Conv2D(32, 32, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            Conv2D(32, 32, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),

            Conv2D(32, 64, kernel_size=3, stride=(3, 1), padding=1, batchnorm=True),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),

            Conv2D(64, 128, kernel_size=3, stride=3, padding=1, batchnorm=True),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),

            Conv2D(128, 256, kernel_size=3, stride=(3, 2), padding=1, batchnorm=True),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True, batchnorm=True),

            Conv2D(256, 512, kernel_size=3, stride=1, padding=0, batchnorm=True),
            Conv2D(512, 512, kernel_size=1, stride=1, padding=0, batchnorm=True),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding





class SyncNetColor(nn.Module):
    def __init__(self):
        super(SyncNetColor, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2D(15, 32, kernel_size=(7, 7), stride=1, padding=3, batchnorm=True),
            # 48, 96 

            Conv2D(32, 64, kernel_size=5, stride=(1, 2), padding=1, batchnorm=True),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            # 48, 48 

            Conv2D(64, 128, kernel_size=3, stride=2, padding=1, batchnorm=True),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            # 24, 24 

            Conv2D(128, 256, kernel_size=3, stride=2, padding=1, batchnorm=True),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            # 12, 12

            Conv2D(256, 512, kernel_size=3, stride=2, padding=1, batchnorm=True),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),
            # 6, 6

            Conv2D(512, 512, kernel_size=3, stride=2, padding=1, batchnorm=True),
            # 3, 3
            Conv2D(512, 512, kernel_size=3, stride=1, padding=0, batchnorm=True),
            # 1, 1, 
            Conv2D(512, 512, kernel_size=1, stride=1, padding=0, batchnorm=True),) # 1, 1
        
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
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, batchnorm=True, residual=True),

            Conv2D(256, 512, kernel_size=3, stride=1, padding=0, batchnorm=True),
            Conv2D(512, 512, kernel_size=1, stride=1, padding=0,  batchnorm=True),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding
