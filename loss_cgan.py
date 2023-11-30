import torch.nn as nn
import torch 

class adversarial_loss(nn.Module):
    def __init__(self):
        super(adversarial_loss, self).__init__()

        self.loss1 = nn.L1Loss()
        self.loss2 = nn.BCELoss()

    def forward(self, real_speech, enhanced_speech, real_output, fake_output):
        loss1 = self.loss1(enhanced_speech, real_speech)

        real_output = torch.mean(real_output)
        fake_output = torch.mean(fake_output)

        loss2 = self.loss2(real_output, torch.ones_like(real_output)) + self.loss2(fake_output, torch.zeros_like(fake_output))
        return loss1 + loss2

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = adversarial_loss()
    loss = loss.to(device)
    input1 = torch.randn(20, 1, 80, 80).to(device)
    input2 = torch.randn(20, 1, 80, 80).to(device)
    input3 = torch.randn(20, 1).to(device)
    input4 = torch.randn(20, 1).to(device)
    _loss = loss(input1, input2, input3, input4)
    print(_loss.shape)

if __name__ == '__main__':
    # test()
    pass
