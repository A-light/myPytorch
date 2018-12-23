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



class Gmnnet(nn.Module):
	"""docstring for ClassName"""
	def __init__(self):
		super(Gmnnet, self).__init__()
		self.conv1 = nn.Conv2d(1,64,3,1,1)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(64,128,3,1,1)
		self.pooling = nn.MaxPool2d(2,2)
		self.fc = nn.Sequential(
				nn.Linear(14*14*128,1024),
				nn.ReLU(),
				nn.Dropout(0.2),
				nn.Linear(1024,10),

			)

	def forward(self,x):
		x=self.conv1(x)
		x=self.relu(x)
		x=self.conv2(x)
		x=self.pooling(x)
		out=x.view(x.size()[0],-1)
		out=self.fc(out)
		return out
		
mynet=Gmnnet()

criteron=nn.CrossEntropyLoss()
optimizer=optim.Adam(mynet.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
	for i,(images,labels) in enumerate(train_loader):
		imagess =images.to(device)
		outputs = mynet(images)
		loss=criteron(outputs,labels)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (epoch+1)%100==0:
			print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
with torch.no_grad():
	correct=0
	total=0
	for images, labels in test_loader:
		images=images.to(device)
		outputs=mynet(images)
		_,predicted=torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()

	print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

torch.save(mynet.state_dict(), 'gmnnet.ckpt')	

