import torch
import torchsummary as summary
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchsummary as summary
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        # Three convolutional blocks
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)

        # Patch-wise prediction layer (5x5)
        self.patch_pred = nn.Conv2d(256, 1, kernel_size=5, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Patch-wise predictions
        x = self.patch_pred(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.sigmoid(x).squeeze()

        return x

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Discriminator(in_channels=2).to(device)
    input = torch.randn(1, 2, 80, 80).to(device)
    summary.summary(model, input_size=(2, 80, 80))
    print(model(input).shape)

if __name__ == "__main__":
    test()
    # pass