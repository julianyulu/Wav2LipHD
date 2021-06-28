import torch
from torch import nn
from torch.nn import functional as F
from .layers.conv import Conv2D, Conv1D

class LdmkSync(nn.Module):
    def __init__(self):
        super(LdmkSync, self).__init__()
        self.audio_encoder = nn.Sequential(
            Conv2D(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2D(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2D(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2D(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2D(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2D(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2D(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2D(512, 512, kernel_size=1, stride=1, padding=0),)

        self.ldmk_encoder = nn.Sequential(
            Conv1D(5, 32, kernel_size = 5, stride = 1, padding = 2),
            Conv1D(32, 32, kernel_size = 3, stride = 1, padding = 1, residual = True),
            Conv1D(32, 32, kernel_size = 3, stride = 1, padding = 1, residual = True), # (B, 32, 74)
            
            Conv1D(32, 64, kernel_size = 3, stride = 2, padding = 1),
            Conv1D(64, 64, kernel_size = 3, stride = 1, padding = 1, residual = True),
            Conv1D(64, 64, kernel_size = 3, stride = 1, padding = 1, residual = True), # (B, 64, 37)
            
            Conv1D(64, 128, kernel_size = 3, stride = 2, padding = 1),
            Conv1D(128, 128, kernel_size = 3, stride = 1, padding = 1, residual = True),
            Conv1D(128, 128, kernel_size = 3, stride = 1, padding = 1, residual = True), # (B, 128, 18)


            Conv1D(128, 256, kernel_size = 3, stride = 2, padding = 1),
            Conv1D(256, 256, kernel_size = 3, stride = 1, padding = 1, residual = True),
            Conv1D(256, 256, kernel_size = 3, stride = 1, padding = 1, residual = True),# (B, 256, 9)

            Conv1D(256, 512, kernel_size = 3, stride = 2, padding = 1),
            Conv1D(512, 512, kernel_size = 3, stride = 1, padding = 1, residual = True),# (B, 512, 5)

            Conv1D(512, 512, kernel_size = 3, stride = 2, padding = 1),
            Conv1D(512, 512, kernel_size = 3, stride = 1, padding = 1, residual = True),
            Conv1D(512, 512, kernel_size = 3, stride = 1, padding = 1, residual = True),# (B, 512, 3)

            Conv1D(512, 512, kernel_size = 3, stride = 1, padding = 0),
            # (B, 512, 1)
        )
            

    def forward(self, audio_sequences, ldmk_sequences):
        ldmk_embedding = self.ldmk_encoder(ldmk_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)
        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        ldmk_embedding = ldmk_embedding.view(ldmk_embedding.size(0), -1)
        audio_embedding = F.normalize(audio_embedding, p = 2, dim = 1)
        ldmk_embedding = F.normalize(ldmk_embedding, p = 1, dim = 1)

        return audio_embedding, ldmk_embedding

            
