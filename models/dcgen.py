import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.parallel

class DCGenerator(nn.Module):
    def __init__(self, z=100, c=3, ngf=64):
        '''
          z: latent variable vector size
          c: number of channels
          ngf: size of generator features
        '''
        super(DCGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z, ngf * 8, 4, 1, 0, bias=False), 
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, c, 4, 2, 1, bias=False),
            nn.Tanh()
         )

    def forward(self, input):
        if input.is_cuda:
            output = nn.parallel.data_parallel(self.model, input)
        else:
            output = self.model(input)
        return output
