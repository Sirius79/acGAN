import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.parallel

class DCDiscriminator2(nn.Module):
  '''
    c: number of channels
    ndf: size of discriminator features
    DCDiscriminator2(
      (main): Sequential(
        (0): Conv2d(1, 64, kernel_size=(6, 6), stride=(2, 2), padding=(1, 1), bias=False)
        (1): LeakyReLU(negative_slope=0.2, inplace)
        (2): Conv2d(64, 128, kernel_size=(6, 6), stride=(2, 2), padding=(1, 1), bias=False)
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): LeakyReLU(negative_slope=0.2, inplace)
        (5): Conv2d(128, 256, kernel_size=(6, 6), stride=(2, 2), padding=(1, 1), bias=False)
        (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): LeakyReLU(negative_slope=0.2, inplace)
        (8): Conv2d(256, 1, kernel_size=(6, 6), stride=(1, 1), bias=False)
        (9): Sigmoid()
      )
    )
  '''
  def __init__(self, c=3, ndf=64):
    super(DCDiscriminator2, self).__init__()
    self.model = nn.Sequential(
        nn.Conv2d(c, ndf, 6, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 6, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 6, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, 1, 6, 1, 0, bias=False),
        nn.Sigmoid()
    )

  def forward(self, input):
    if input.is_cuda > 1:
        output = nn.parallel.data_parallel(self.model, input)
    else:
        output = self.model(input)

    return output.view(-1, 1).squeeze(1)
