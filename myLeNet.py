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

                                          
class MyLeNet(nn.Module):
	"""docstring for MyLeNet"""
	def __init__(self):
		super(MyLeNet,self).__init__()
		self.choice_conv = nn.ModuleDict({
				'conv1':nn.Conv2d(1,10,3,1,2),
				'conv2':nn.Conv2d(10,10,3,1),
				'conv3':nn.Conv2d(1,10,3,1),
				'conv4':nn.Conv2d(1,10,5,1),
				'conv5':nn.Conv2d(1,6,5,1),##c1
				'conv6':nn.Conv2d(6,16,5,1), ##c3
				'conv7':nn.Conv2d(16,120,5,1)
			})
		self.choice_pooling = nn.ModuleDict({
				'maxpooling1':nn.MaxPool2d(2,2),
				#'maxpooling2':nn.MaxPool2d(1,1),
				'avgpooling1':nn.AvgPool2d(2,2),

			})
		self.choice_activations = nn.ModuleDict({
				'rule':nn.ReLU(),
				'leakyrule':nn.LeakyReLU(),
				'logsigmoid':nn.LogSigmoid(),
				'prelu':nn.PReLU(),
				'sigmoid':nn.Sigmoid(),
				'tanh':nn.Tanh(),
				'softmin':nn.Softmin(),
				'softmax':nn.Softmax(),
				'softmax2':nn.Softmax2d()
			})

		self.choice_fc = nn.ModuleDict({
				'f1':nn.Linear(120,84),
				'f2':nn.Linear(84,10)
			})
	
	def forward(self,x):
		x=self.choice_conv['conv5'](x)
		x=self.choice_activations['rule'](x)
		x=self.choice_conv['conv6'](x)
		x=self.choice_activations['rule'](x)
		x=self.choice_pooling['maxpooling1'](x)
		x=self.choice_pooling['maxpooling1'](x)

		x=self.choice_conv['conv7'](x)
		x=self.choice_activations['rule'](x)
		out=x.view(x.size()[0],-1)
		out=self.choice_fc['f1'](out)
		out=self.choice_fc['f2'](out)
		return out


mylenet=MyLeNet()

criteron=nn.CrossEntropyLoss()
optimizer=optim.SGD(mylenet.parameters(),lr=learning_rate)


for i in range(num_epochs):
	for j,(x,y_) in enumerate(train_loader):
		images = x.to(device)
		y_ = y_.to(device)
		y = mylenet.forward(images)
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
		images=images.to(device)
		labels=labels.to(device)

		y = mynet.forward(images)
		
		_,predicted=torch.max(y.data,1)
		total+=labels.size(0) #??
		correct+=(predicted==labels).sum() #??
	
	print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

torch.save(mylenet.state_dict(), 'MyLeNet.ckpt')


