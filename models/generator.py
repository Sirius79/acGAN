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

class Generator(nn.Module):
  def __init__(self, nz):
    super(Generator, self).__init__()
    self.main = nn.Sequential(
        nn.Linear(nz, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 512),
        nn.LeakyReLU(0.1),
        nn.Linear(512, 1024),
        nn.LeakyReLU(0.1),
        nn.Linear(1024, i),
        nn.Tanh()
    )
 
  def forward(self, input):
    if input.is_cuda:
      output = nn.parallel.data_parallel(self.main, input)
    else:
      output = self.main(input)
    return output
