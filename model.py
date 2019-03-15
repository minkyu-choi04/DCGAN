import torch
import torch.nn as nn

def init_weights(m):
    'https://pytorch.org/docs/stable/nn.html#torch.nn.Module.apply'
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    else:
        pass

class Generator(nn.Module):
    def __init__(self, z_dim=64, img_c=1, fm_c=[1024, 512, 256, 128]):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.img_c = img_c
        self.fm_c = [z_dim] + fm_c + [img_c]
        self.generator = []
        for i in range(len(self.fm_c)-1):
            if i == 0:
                self.generator += [nn.ConvTranspose2d(self.fm_c[i], self.fm_c[i+1], kernel_size=4, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(self.fm_c[i+1]), 
                        nn.ReLU(),]
            elif i == len(self.fm_c)-2:
                self.generator += [nn.ConvTranspose2d(self.fm_c[i], self.fm_c[i+1], kernel_size=4, stride=2, padding=1, bias=False), 
                        nn.Tanh(),]
            else:
                self.generator += [nn.ConvTranspose2d(self.fm_c[i], self.fm_c[i+1], kernel_size=4, stride=2, padding=1, bias=False), 
                        nn.BatchNorm2d(self.fm_c[i+1]), 
                        nn.ReLU(),]
        self.generator = nn.Sequential(*self.generator)

    def forward(self, input):
        return self.generator(input)

class Discriminator(nn.Module):
    def __init__(self, img_c=1, fm_c=[128, 256, 512, 1024]):
        super(Discriminator, self).__init__()
        self.img_c = img_c
        self.fm_c = [img_c] + fm_c + [1]
        self.discriminator = []
        for i in range(len(self.fm_c)-1):
            if i == 0: 
                self.discriminator += [nn.Conv2d(self.fm_c[i], self.fm_c[i+1], kernel_size=4, stride=2, padding=1, bias=False), 
                        nn.LeakyReLU(0.2, inplace=False),]
            elif i != len(self.fm_c)-2: 
                self.discriminator += [nn.Conv2d(self.fm_c[i], self.fm_c[i+1], kernel_size=4, stride=2, padding=1, bias=False), 
                        nn.BatchNorm2d(self.fm_c[i+1]),
                        nn.LeakyReLU(0.2, inplace=False),]
            else:
                self.discriminator += [nn.Conv2d(self.fm_c[i], self.fm_c[i+1], kernel_size=4, stride=1, padding=0, bias=False),
                        nn.Sigmoid()]
        self.discriminator = nn.Sequential(*self.discriminator)


    def forward(self, input):
        return self.discriminator(input)


