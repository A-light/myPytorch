import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.datasets import CIFAR10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 5
batch_size = 100

train_dataset = torchvision.datasets.CIFAR10(root='/data_cifar10',
											train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)
test_dataset = torchvision.datasets.CIFAR10(root='/data_cifar10', 
											train=False, 
											transform=transforms.ToTensor(),
											download=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
											 batch_size,
											 shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
											batch_size,
										 shuffle=False)

class GoogLeNetI(object):
	"""docstring for GoogLeNetI"""
	def __init__(self, arg):
		super(GoogLeNetI, self).__init__()
		self.arg = arg
		