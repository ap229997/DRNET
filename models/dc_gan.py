import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from basic_block import *

'''
Encoder network based on DC-GAN architecture
'''
class Encoder(nn.Module):
	'''
	Arguments:
		in_channels : no. of channels in the input (can be either grayscale(1) or color image(3))
		out_channels : no. of channels in the feature vector (can be either content or pose features)
		normalize : whether to normalize the feature vector
	'''
	def __init__(self, in_channels, out_channels, normalize=False):
		super(Encoder, self).__init__()

		self.normalize = normalize

		self.block1 = BasicConv(in_channels, 64, 4, 2, 1)

		self.block2 = BasicConv(64, 128, 4, 2, 1)

		self.block3 = BasicConv(128, 256, 4, 2, 1)

		self.block4 = BasicConv(256, 512, 4, 2, 1)

		self.block5 = BasicConv(512, 512, 4, 2, 1)

		self.block6 = nn.Sequential(
						nn.Conv2d(512, out_channels, 4, 1, 0),
						nn.BatchNorm2d(out_channels),
						nn.Tanh(),
						)

	'''
	Computes a forward pass through the specified architecture

	Arguments:
		inp : input (generally image in grayscale or color form)

	Returns:
		out6 : output of the forward pass applied to the input image
		[out1, out2, out3, out4, out5] : outputs at each stage which may be used
			   if skip connection functionality is included in the architecture

	Note : The encoder is coded so as to return skip connections at each stage everytime.
		   These skip connections can be used in the decoder stages if required.
	'''
	def forward(self, inp):
		out1 = self.block1(inp)
		out2 = self.block2(out1)
		out3 = self.block3(out2)
		out4 = self.block4(out3)
		out5 = self.block5(out4)
		out6 = self.block6(out5)

		if self.normalize:
			out6 = F.normalize(out6, p=2)

		return out6, [out1, out2, out3, out4, out5]

'''
Decoder based on DC-GAN architecture
'''
class Decoder(nn.Module):
	'''
	Arguments:
		in_channels : no. of channels in the input feature vector 
					  (generally concatenation of content and pose features)
		out_channels : no. of channels in the output (generally the original
					   image dimension - 1 for grayscale or 3 for color)
		use_skip : whether to use the skip connection functionality
	'''
	def __init__(self, in_channels, out_channels, use_skip=False):
		super(Decoder, self).__init__()

		self.use_skip = use_skip
		# if the skip connections are used, then the input at each stage is the
		# concatenation of current feature and feature vector from the encoder
		# hence double the channels, so mul_factor (multiplication factor) is 
		# used to incorporate this effect
		self.mul_factor = 1
		if self.use_skip:
			self.mul_factor = 2

		self.block1 = BasicConvTranspose(in_channels, 512, 4, 1, 0)

		self.block2 = BasicConvTranspose(512*self.mul_factor, 512, 4, 2, 1)

		self.block3 = BasicConvTranspose(512*self.mul_factor, 256, 4, 2, 1)
	
		self.block4 = BasicConvTranspose(256*self.mul_factor, 128, 4, 2, 1)
	
		self.block5 = BasicConvTranspose(128*self.mul_factor, 64, 4, 2, 1)

		self.block6 = nn.Sequential(
					nn.ConvTranspose2d(64*self.mul_factor, out_channels, 4, 2, 1),
					nn.Sigmoid(),
					)
	'''
	Computes a forward pass through the specified decoder architecture

	Arguments:
		content : content feature vector
		skip : skip connections (used only if requried)
		pose : pose feature vector

	Returns:
		out6 : result of the forward pass (generally the same dimension as the original image)
	'''
	def forward(self, content, skip, pose):

		inp1 = torch.cat([content, pose], dim=1)
		out1 = self.block1(inp1)

		# if skip connections are to be used, then the input at each stage
		# is the concatenation of current feature vector and the skip 
		# connection feature vector from the encoder
		if self.use_skip:
			inp2 = torch.cat([out1, skip[4]], dim=1)
		else:
			inp2 = out1
		out2 = self.block2(inp2)

		if self.use_skip:
			inp3 = torch.cat([out2, skip[3]], dim=1)
		else:
			inp3 = out2
		out3 = self.block3(inp3)

		if self.use_skip:
			inp4 = torch.cat([out3, skip[2]], dim=1)
		else:
			inp4 = out3
		out4 = self.block4(inp4)

		if self.use_skip:
			inp5 = torch.cat([out4, skip[1]], dim=1)
		else:
			inp5 = out4
		out5 = self.block5(inp5)

		if self.use_skip:
			inp6 = torch.cat([out5, skip[0]], dim=1)
		else:
			inp6 = out5
		out6 = self.block6(inp6)

		return out6

# testing code
if __name__ == '__main__':
	torch.cuda.set_device(0)
	net = Encoder(3,128).cuda()
	inp = Variable(torch.FloatTensor(4,3,256,256)).cuda()
	out, skip = net(inp)
	print ('content after encoder: ', out.shape)

	net = Encoder(3,10,normalize=True).cuda()
	pose, _ = net(inp)
	print ('pose after encoder: ', pose.shape)

	net = Decoder(128+10,3,use_skip=True).cuda()
	out = net(out, skip, pose)
	print ('out after decoder: ', out.shape)