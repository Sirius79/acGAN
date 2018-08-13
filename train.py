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
import random
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F

def train(G, D_set, optimG, optimD, dataloader, criterion, N, T_max, T_w=None, alpha=0.1, _lambda=15):
  '''
    G: generator
    D_set: set of discriminators
    optimG: generator optimiser
    optimD: set of discriminator optimiser
    dataloader: dataloader
    criterion: loss function
    T_max: max time steps
    T_w: warm up time
    alpha:moving average coefficient
    _lambda: Boltzmann constant
  '''
  if T_w is None:
    T_w = 10 * N 
    
  # initialize prob vector
  pi = torch.full((N, ), 1/N, device=device)
  # initialize Q values
  Q = torch.zeros(N, device=device)

  losses_D, losses_G = [[],[],[]], []

  for t in range(T_max):
    
    G_prev = G
    for i, data in enumerate(dataloader, 0):
      
      # real and fake labels
      real_label = random.uniform(0.7, 1.2)
      fake_label = random.uniform(0.0, 0.3)
      
      real_data = data[0].to(device)
      batch_size = real_data.size(0)
      label = torch.full((batch_size,), real_label, device=device)
      
      # train D_set using 3
      for j, D in enumerate(D_set):
        
        # train with real
        D.zero_grad()
        output = D(real_data)
        errD_real = pi[j] * criterion(output, label)
        errD_real.backward()
        
        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = G(noise)
        label.fill_(fake_label)
        output = D(fake.detach())
        errD_fake = pi[j] * criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        losses_D[j].append(errD)
        optimD[j].step()
      
      # train G using 2
      label.fill_(real_label)  # fake labels are real for generator cost
      total_error = 0
      for j, D in enumerate(D_set):
        G.zero_grad()
        output = D(fake)
        errG = pi[j] * criterion(output, label)
        total_error += errG
      errG.backward()
      losses_G.append(errG)
      optimG.step()
      print('Done: [%d/%d][%d/%d]' % (t, T_max, i, len(dataloader)))
      
    if t >= T_w:
      
      real_label = random.uniform(0.7, 1.2)
      label = torch.full((batch_size,), real_label, device=device)
      
      # choose arm k according to 7
      k = np.random.choice(N, 1, p=pi)
      
      # evaluate performance of G with D_k
      noise = torch.randn(batch_size, nz, 1, 1, device=device)
      fake = G(noise)
      output = D_set[k](fake)
      
      # receive reward among 4, 5
      output_prev_G = D_set[k](G_prev(noise))
      reward = criterion(output, label) - criterion(output_prev_G, label)
      
      # update Q and pi values according to 6, 7
      Q[k] = alpha * reward + (1 - alpha) * Q[k]
      pi = F.softmax(_lambda * Q)
      
  return losses_D, losses_G
    
