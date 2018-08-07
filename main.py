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

# add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | mnist')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')

# check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = parser.parse_args()

# load data
d = 'mnist'
dr = '../../data/'
bs = 64
i = 784

if opt.dataset == 'mnist':
  dataset = dset.MNIST(root='../../data/', download=True, train=True,
                       transform=transforms.Compose([transforms.Resize(opt.imageSize),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
elif opt.dataset == 'cifar10':
  dataset = dset.CIFAR10(root='../../data/', download=True, train=True,\
                         transform=transforms.Compose([transforms.Resize(opt.imageSize),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

# latent vector size
nz = int(opt.nz)

# noise
fixed_noise = torch.randn(opt.batchSize, opt.nz, device=device)

# intialize G and D
G = Generator().to(device)
D = Discriminator().to(device)

# one pass
op = G(fixed_noise)
out = D(op)
