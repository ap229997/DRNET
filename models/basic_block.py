import torch
import torch.nn as nn
from torch.autograd import Variable

'''
Basic convolution block consisting of convolution layer, batch norm and leaky relu
'''
class BasicConv(nn.Module):
	'''
	Arguments (basic parameters of a convolution layer):
		in_channels : input planes
		out_channels : output planes
		kernel_size : size of filter to be applied
		stride : striding to be used
		padding : padding to be used
	'''
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
		super(BasicConv, self).__init__()

		self.basic_block = nn.Sequential(
							nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
										stride=stride, padding=padding),
							nn.BatchNorm2d(out_channels),
							nn.LeakyReLU(0.2, inplace=True),
							)
	'''
	Computes a forward pass through the basic conv block

	Arguments:
		inp : input feature

	Return:
		out : output of the forward pass applied to input feature
	'''
	def forward(self, inp):
		out = self.basic_block(inp)
		return out

'''
Basic transpose convolution block consisting of transpose convolution layer, batch norm and leaky relu
'''
class BasicConvTranspose(nn.Module):
	'''
	Arguments (basic parameters of a transpose convolution layer):
		in_channels : input planes
		out_channels : output planes
		kernel_size : size of filter to be applied
		stride : striding to be used
		padding : padding to be used
	'''
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
		super(BasicConvTranspose, self).__init__()

		self.basic_block = nn.Sequential(
							nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
										stride=stride, padding=padding),
							nn.BatchNorm2d(out_channels),
							nn.LeakyReLU(0.2, inplace=True),
							)

	'''
	Computes a forward pass through the basic transpose conv block

	Arguments:
		inp : input feature

	Return:
		out : output of the forward pass applied to input feature
	'''
	def forward(self, inp):
		out = self.basic_block(inp)
		return out

# testing code
if __name__ == '__main__':
	torch.cuda.set_device(0)
	net = BasicConv(3,64).cuda()
	inp = Variable(torch.FloatTensor(4,3,256,256)).cuda()
	out = net(inp)
	print ('out: ', out.shape)

	net = BasicConvTranspose(3,512,4,1,0).cuda()
	inp = Variable(torch.FloatTensor(4,3,256,256)).cuda()
	out = net(inp)
	print ('out: ', out.shape)
