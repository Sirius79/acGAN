import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.parallel
import argparse
import matplotlib.pyplot as plt
import numpy as np
from models import *
from train import train

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | mnist')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--z', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='size of the generator features')
parser.add_argument('--ndf', type=int, default=64, help='size of the discriminator features')
parser.add_argument('--n', type=int, default=3, help='number of discriminators')
parser.add_argument('--ts', type=int, default=25, help='number of time steps to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0001')

# check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = parser.parse_args()

# load data
if opt.dataset == 'mnist':
  dataset = dset.MNIST(root='../../data/', download=True, train=True,
                       transform=transforms.Compose([transforms.Resize(opt.imageSize),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
  ch = 1
elif opt.dataset == 'cifar10':
  dataset = dset.CIFAR10(root='../../data/', download=True, train=True,\
                         transform=transforms.Compose([transforms.Resize(opt.imageSize),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
  ch = 3

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

# vector sizes
z = int(opt.z)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
T_max = int(opt.ts)
N = int(opt.n)

# intialization
DCG = DCGenerator(c=ch).to(device)
DCG.apply(weights_init)

criterion = nn.BCELoss()
optimizerG = optim.Adam(DCG.parameters(), lr=opt.lr)

if opt.n == 3:
  DCD1 = DCDiscriminator1(c=ch).to(device)
  DCD1.apply(weights_init)
  
  DCD2 = DCDiscriminator2(c=ch).to(device)
  DCD2.apply(weights_init)
  
  DCD3 = DCDiscriminator3(c=ch).to(device)
  DCD3.apply(weights_init)
  
  optimizerD1 = optim.Adam(DCD1.parameters(), lr=opt.lr)
  optimizerD2 = optim.Adam(DCD2.parameters(), lr=opt.lr)
  optimizerD3 = optim.Adam(DCD3.parameters(), lr=opt.lr)

loss_d, loss_g = train(DCG, [DCD1, DCD2, DCD3], optimizerG, [optimizerD1, optimizerD2, optimizerD3],\
                       dataloader, criterion, z, N, T_max, 5)
