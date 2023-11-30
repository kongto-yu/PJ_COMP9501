from generator import Generator
from discriminator import Discriminator
import torch
import torch.nn as nn
import torchsummary as summary

class cGAN(nn.Module):
    def __init__(self, in_channels_g, in_channels_d, out_channels_g, bilinear=False):
        super(cGAN, self).__init__()
        self.generator = Generator(in_channels_g, out_channels_g, bilinear)
        self.discriminator = Discriminator(in_channels_d)

    def forward(self, mmwave, real_speech):
        enhanced_speech = self.generator(mmwave)

        real_input = torch.cat((mmwave, real_speech), 1)
        fake_input = torch.cat((mmwave, enhanced_speech), 1)

        real_output = self.discriminator(real_input)
        fake_output = self.discriminator(fake_input)

        return enhanced_speech, real_output, fake_output
    

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cGAN(in_channels_g=1, in_channels_d=2, out_channels_g=1, bilinear=False)
    model = model.to(device)
    mmwave = torch.randn(1, 1, 80, 80).to(device)
    
    real_speech = torch.randn(1, 1, 80, 80).to(device)
    enhanced_speech, real_output, fake_output = model(mmwave, real_speech)

    print(enhanced_speech.shape, real_output.shape, fake_output.shape)
    summary.summary(model, [(1, 80, 80), (1, 80, 80)])


if __name__ == '__main__':
    # test()
    pass