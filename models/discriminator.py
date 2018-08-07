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

class Discriminator(nn.Module):
  def __init__(self, imageSize):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
        nn.Linear(imageSize, 1024),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.3),
        nn.Dropout(0.4),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
  
  def forward(self, input):
    if input.is_cuda:
      output = nn.parallel.data_parallel(self.main, input)
    else:
      output = self.main(input)
    return output
