import torch.nn as nn
from .layers.conv import Conv2D, Conv3D

import pdb

class SyncNet(nn.Module):
    # MFCC model
    def __init__(self, feat_dim = 1024):
        super(SyncNet, self).__init__()
        self.audio_encoder = nn.Sequential(
            Conv2D(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.MaxPool2d(kernel_size=(1,1), stride=(1,1)),

            Conv2D(64, 192, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,2)),

            Conv2D(192, 384, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            Conv2D(384, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            
            Conv2D(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

            Conv2D(256, 512,  kernel_size=(5,4), stride=(1,1), padding=(0,0)),
        )
        
        self.face_encoder = nn.Sequential(
            Conv3D(3, 96, kernel_size=(5,7,7), stride=(1,2,2), padding=0),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),
            
            Conv3D(96, 256, kernel_size=(1,5,5), stride=(1,2,2), padding=(0,1,1)),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),

            Conv3D(256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            Conv3D(256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            
            Conv3D(256, 256, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1)),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2)),

            Conv3D(256, 512, kernel_size=(1,6,6), stride=(1,1,1), padding=0)
        )

        
        self.audio_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feat_dim),
        )

        self.face_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, feat_dim),
        )

        
    def forward(self, audio_sequence, face_sequence):
        """
        face_sequence: # (B, 3, 5, 240, 240)
        audio_sequence: # (B, 1, 13, 20)
        """
        audio_feats = self.audio_encoder(audio_sequence) # (B, 512, 1, 1)
        audio_feats = audio_feats.view((audio_feats.size(0), -1)) #(B, 512)
        audio_feats = self.audio_fc(audio_feats) # (B, 1024)

        face_feats = self.face_encoder(face_sequence)  # (B, 512, 1, 1, 1)
        face_feats = face_feats.view((face_feats.size(0), -1)) # (B, 512)
        face_feats = self.face_fc(face_feats) #(B, 1024)

        return audio_feats, face_feats

    def forward_facefeat(self, x):
        face_feats = self.face_encoder(face_sequence)
        face_feats = face_feats.view((face_feats.size(0), -1)) # (N, (ch*24))
        return face_feats 
