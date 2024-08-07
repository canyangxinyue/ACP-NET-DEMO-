import torch.nn as nn

class Empty(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels=kwargs.get('in_channels',256)
        self.out_channels=self.in_channels

    def forward(self, x):
        return x
