import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
Discriminator to discriminate between two poses.
The pose features shouldn't contain any content
information and using the discriminator ensures
that the pose of any two frames from the same 
video or from different videos remain indistinguishable
ensuring that they are independent of any content
information

The Scene Discriminator is modelled using a two layer
MLP with the specifications described below
'''
class SceneDiscriminator(nn.Module):
	'''
	Arguments:
		pose_channels : no. of channels in the pose features
		hidden_units : represent the no. of neurons in the
					   hidden layer

	Note : pose_channels*2 is used in line 36 since we need to
		   provide the two poses which need to be distinguished,
		   hence the input dimension is twice
	'''
	def __init__(self, pose_channels, hidden_units=100):
		super(SceneDiscriminator, self).__init__()

		self.pose_channels = pose_channels
		self.hidden_units = hidden_units

		self.fc = nn.Sequential(
					nn.Linear(self.pose_channels*2, self.hidden_units),
					nn.ReLU(inplace=True),
					nn.Linear(self.hidden_units, self.hidden_units),
					nn.ReLU(inplace=True),
					nn.Linear(self.hidden_units, 1),
					nn.Sigmoid(),
					)
	'''
	Computes a forward pass through the discriminator

	Arguments:
		pose1, pose2 : the two poses among which to differentiate

	Returns:
		out : output of the forward pass applied to concatenation of two poses
	'''
	def forward(self, pose1, pose2):

		pose = torch.cat([pose1, pose2], dim=1) # concatenating the two poses
		pose = pose.view(-1, self.pose_channels*2)

		out = self.fc(pose)

		return out

# testing code
if __name__ == '__main__':
	torch.cuda.set_device(0)
	net = SceneDiscriminator(10,100).cuda()
	pose1 = Variable(torch.FloatTensor(3,10,5,5)).cuda()
	pose2 = Variable(torch.FloatTensor(3,10,5,5)).cuda()
	out = net(pose1, pose2)
	print ('out: ', out.shape)