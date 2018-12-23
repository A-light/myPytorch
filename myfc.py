import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import numpy as np 
import matplotlib.pyplot  as plot
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size=784
hidden_size=500
output_size=10

num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer


class MyNet(nn.Module):
	"""docstring for MyNet"""
	def __init__(self):
		super(MyNet, self).__init__()
		self.layer1 = nn.Linear(input_size,hidden_size)
		self.layer2 = nn.Linear(hidden_size,output_size)
	
	def forward(self,x):
		x=self.layer1(x)
		x=F.relu(x)
		return self.layer2(x)

	# def criteron(y,y_):
	# 	crn=nn.CrossEntropyLoss()
	# 	return crn(y,y_)

	# def optimizer():
	# 	return optim.SGD(self.parameters(),lr=learning_rate)

mynet=MyNet()
total_step = len(train_loader)

criteron=nn.CrossEntropyLoss()
optimizer=optim.SGD(mynet.parameters(),lr=learning_rate)

for i in range(num_epochs):
	for j,(x,y_) in enumerate(train_loader):
		images = x.reshape(-1,28*28).to(device)
		y_ = y_.to(device)
		y = mynet.forward(images)
		loss = criteron(y,y_)

		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1)%100 ==0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
				.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

with torch.no_grad():
	correct=0
	total=0
	for images,labels in test_loader:
		images = images.reshape(-1,28*28).to(device)
		labels=labels.to(device)

		y = mynet.forward(images)
		
		_,predicted=torch.max(y.data,1)
		total+=labels.size(0)
		correct+=(predicted==labels).sum()
	
	print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

torch.save(mynet.state_dict(), 'myfc.ckpt')


