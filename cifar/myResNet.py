import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets import CIFAR10

class ResNetI(nn.Module):
	"""docstring for ResNetI"""
	def __init__(self, arg):
		super(ResNetI, self).__init__()
		self.conv1 = nn.Conv2d(1,)

	def Conv3x3(self,input_channels,output_chanels,n):
		for i in range(n):
			x=nn.Conv2d(input_channels,output_chanels,3,1,1)

		return x+
