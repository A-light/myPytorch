import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.datasets import CIFAR10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 5
batch_size = 100
learning_rate = 0.001

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

def Conv3x3(input_channels,output_chanels,n):#1,2
		current_layer = nn.Sequential(
			nn.Conv2d(input_channels,output_chanels,3,1),
			nn.ReLU()
		)
		for i in range(n-1):
			input_channels = output_chanels
			output_chanels = 2*output_chanels
			print(input_channels,output_chanels)
			conv = nn.Conv2d(input_channels,output_chanels,3,1)
			current_layer.add_module('inner_layer{}'.format(i),conv)
		return current_layer
def data_reshape(input_channels,output_chanels,x):
	conv=nn.Conv2d(input_channels,output_chanels,5,1)
	out=conv(x)
	return out

def data_reshape1(input_channels,output_chanels,x):
	conv=nn.Linear(input_channels,output_chanels,5,1)
	out=conv(x)
	return out


class ResNetI(nn.Module):
	"""docstring for ResNetI"""
	def __init__(self):
		super(ResNetI, self).__init__()
		#self.layer = nn.Conv2d(1,64,7,2)
		self.layer0 = nn.Sequential(
				nn.Conv2d(3,2,3,1,1),
				nn.ReLU()
			)
		self.layer1 = Conv3x3(2,2,2)
		self.layer2 = Conv3x3(4,4,2)
		self.layer3 = Conv3x3(8,8,2)
		self.layer4 = Conv3x3(16,16,2)
		self.layer5 = Conv3x3(32,32,2)
		self.layer6 = Conv3x3(64,128,2)
		self.pooling = nn.Sequential(
				nn.MaxPool2d(2,2),
				nn.MaxPool2d(2,2),
				nn.AvgPool2d(2,2)
			)
		self.fc = nn.Sequential(
				nn.Linear(128*2,10),
				nn.ReLU()
			)



	def forward(self,x):
		#print(x.size())
		x = self.layer0(x)

		out = self.layer1(x)
		#print(x.size(),out.size())
		#x=nn.BatchNorm2d(out.size()[1])
		#
		x=data_reshape(x.size()[1],out.size()[1],x)
		#print(x.size(),out.size())
		out = F.relu(x+out)
		x = out

		out = self.layer2(x)
		#print(x.size(),out.size())
		#x=nn.BatchNorm2d(out.size()[1])
		#
		x=data_reshape(x.size()[1],out.size()[1],x)
		#print(x.size(),out.size())
		out = F.relu(x+out)
		x = out
		#print(type(out))

		out = self.layer3(x)
		#print(x.size(),out.size())
		#x=nn.BatchNorm2d(out.size()[1])
		#
		x=data_reshape(x.size()[1],out.size()[1],x)
		#print(x.size(),out.size())
		out = F.relu(x+out)
		x = out

		out = self.layer4(x)
		#print(x.size(),out.size())
		#x=nn.BatchNorm2d(out.size()[1])
		#
		x=data_reshape(x.size()[1],out.size()[1],x)
		#print(x.size(),out.size())
		out = F.relu(x+out)
		x = out

		out = self.layer5(x)
		#print(x.size(),out.size())
		#x=nn.BatchNorm2d(out.size()[1])
		#
		x=data_reshape(x.size()[1],out.size()[1],x)
		#print(x.size(),out.size())
		out = F.relu(x+out)
		x = out

		out = self.layer6(x)
		#print(x.size(),out.size())
		#x=nn.BatchNorm2d(out.size()[1])
		#
		x=data_reshape(x.size()[1],out.size()[1],x)
		#print(x.size(),out.size())
		out = F.relu(x+out)
		x = out
		#print(x.size())

		x=self.pooling(x)
		#print(x.size())
		x=x.view(x.size()[0],-1)
		
		return self.fc(x)
mynet=ResNetI()

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

