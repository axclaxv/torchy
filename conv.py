import torch
import torch.nn as nn

def block(out_channels, activation=None, **kwargs):
    if activation is None:
        activation = nn.ReLU()
    return nn.Sequential(
        nn.LazyConv2d(out_channels, **kwargs),
        nn.BatchNorm2d(out_channels),
        activation
    )

class msblock(nn.Module):
    def __init__(self, out_channels, activation=None, scales=None, **kwargs):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        if scales is None:
            scales = [3, 5, 7]
        self.convs = nn.ModuleList([
            block(out_channels, activation=activation, kernel_size=k, padding='same')
            for k in scales
        ])
        self.merge = block(out_channels, activation=activation, kernel_size=1)

    def forward(self, x):
        y = [x] + [conv(x) for conv in self.convs]
        y = torch.cat(y, dim=-3)
        y = self.merge(y)
        return y