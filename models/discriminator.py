import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """
    Simple CNN discriminator for GoL patterns with feature extraction.
    Returns probability and list of intermediate feature maps.
    """
    def __init__(self, in_channels=1, base_channels=64, channel_mults=(1,2,4)):
        super().__init__()
        layers = []
        last_c = in_channels
        for mult in channel_mults:
            out_c = base_channels * mult
            layers.append(nn.Conv2d(last_c, out_c, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            last_c = out_c
        self.conv = nn.Sequential(*layers)
        # final classification layer
        self.classifier = nn.Conv2d(last_c, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        feats = []
        for layer in self.conv:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(x)
        out = self.classifier(x)
        out = out.view(x.size(0), -1)
        prob = torch.sigmoid(out)
        return prob, feats
