'''
Refere to the paper 'Unsupervised Learning of Disentangled
Representations from Video' - https://arxiv.org/pdf/1705.10915.pdf
for details about the training mechanism and specifications
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import os
import numpy as np
import argparse
import random

import models.vgg as vgg
import models.dc_gan as dc_gan
import models.resnet as resnet
import models.lstm as lstm
import models.scene_discriminator as scene_discriminator

from data_loader import loader

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--content_dim', type=int, default=128, help='no. of channels in content features')
parser.add_argument('--pose_dim', type=int, default=5, help='no. of channels in pose features')
parser.add_argument('--dataset', type=str, default='mnist', help='dataset to use: mnist / kth / suncg / norb')
parser.add_argument('--use_skip', type=bool, default=False, help='whether to use skip connections or not')
parser.add_argument('--resnet_version', type=int, default=18, help='18 / 34 / 50 / 101 / 152')
parser.add_argument('--discriminator_hidden_units', type=int, default=100, help='no. of units in the hidden layer of discriminator')
parser.add_argument('--lstm_hidden_units', type=int, default=256, help='no. of units in the hidden layer of lstm')
parser.add_argument('--layers', type=int, default=1, help='no. of lstm layers')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--save_dir', type=str, default='saved_models', help='directory to save models')
parser.add_argument('--start_iter', type=int, default=0, help='load the saved model from given iteration')
parser.add_argument('--max_iter', type=int, default=50000, help='total no. of iters to train the model')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--data_root', default='', help='root directory for data')
parser.add_argument('--max_step', type=int, default=20, help='maximum distance between frames')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--image_width', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--data_type', default='drnet', help='speed up data loading for drnet training')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--data_threads', type=int, default=5, help='no. of parallel data loading threads')
parser.add_argument('--alpha', type=float, default=1, help='weight for the reconstruction loss')
parser.add_argument('--beta', type=float, default=0.1, help='weight for the discriminator loss')
parser.add_argument('--save_iters', type=int, default=500, help='no. of iterations after which to save the model')
# arguments needed for training lstm
parser.add_argument('--n_past', type=int, default=10, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--use_lstm', type=bool, default=False, help='whether to use lstm')

args = parser.parse_args()

torch.cuda.set_device(args.gpu)

# load the content_encoder, pose_encoder, decoder, scene_discriminator and lstm
# as per the required specifications
if args.dataset == 'mnist':
	content_encoder = dc_gan.Encoder(args.channels, args.content_dim)
	pose_encoder = dc_gan.Encoder(args.channels, args.pose_dim)
	decoder = dc_gan.Decoder(args.content_dim + args.pose_dim, args.channels, args.use_skip)
elif args.dataset == 'suncg' or 'norb':
	content_encoder = resnet.Encoder(args.channels, args.content_dim, args.resnet_version, pretrained=True)
	pose_encoder = resnet.Encoder(args.channels, args.pose_dim, args.resnet_version, pretrained=True)
	decoder = dc_gan.Decoder(args.content_dim + args.pose_dim, args.channels, args.use_skip)
elif args.dataset == 'kth':
	content_encoder = vgg.Encoder(args.channels, args.content_dim)
	pose_encoder = resnet.Encoder(args.channels, args.pose_dim, args.resnet_version, pretrained=True)
	decoder = vgg.Decoder(args.content_dim + args.pose_dim, args.channels, args.use_skip)

scene_discriminator = scene_discriminator.SceneDiscriminator(args.pose_dim, args.discriminator_hidden_units)
lstm = lstm.LSTM(args.content_dim + args.pose_dim, args.lstm_hidden_units, args.pose_dim, args.batch_size, args.layers)

# optimizers for each of the network components
optimizer_content_encoder = optim.Adam(content_encoder.parameters(), lr=args.lr)
optimizer_pose_encoder = optim.Adam(pose_encoder.parameters(), lr=args.lr)
optimizer_decoder = optim.Adam(decoder.parameters(), lr=args.lr)
optimizer_scene_discriminator = optim.Adam(scene_discriminator.parameters(), lr=args.lr)
optimizer_lstm = optim.Adam(lstm.parameters(), lr=args.lr)

# loss functions
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()

# transfer all network components to GPU
content_encoder.cuda()
pose_encoder.cuda()
decoder.cuda()
scene_discriminator.cuda()
lstm.cuda()
mse_loss.cuda()
bce_loss.cuda()

'''
training the discriminator
'''
def train_scene_discriminator(x):
	scene_discriminator.zero_grad()

	target = torch.cuda.FloatTensor(args.batch_size, 1)
	# in accordance with the input to the discriminator
	# two poses are requried, hence two frames are given
	# as input and pose features are computed for both
	x1 = x[0]
	x2 = x[1]
	h_p1 = pose_encoder(x1)[0].detach()
	h_p2 = pose_encoder(x2)[0].detach()

	# randomly select half of the batch and assign zero labels
	# this is required since the discriminator should be able 
	# to distinguish any two frames of a video, hence half the
	# pose features are randomly assigned zero labels and other
	# half is assigned one labels.
	half = int(args.batch_size/2)
	rp = torch.randperm(half).cuda() # randomly select half the batch
	h_p2[:half] = h_p2[rp]
	target[:half] = 1
	target[half:] = 0

	out = scene_discriminator(h_p1, h_p2)
	# compute the bce and accuracy scores as shown
	bce = bce_loss(out, Variable(target))
	acc =out[:half].gt(0.5).sum() + out[half:].le(0.5).sum()

	# run the backprop and update the parameters 
	bce.backward()
	optimizer_scene_discriminator.step()

	# return the bce and accuracy values after taking mean across the batch
	return bce.data/args.batch_size, acc.data/args.batch_size

'''
training the main network consisting of content encoder, pose encoder and decoder
'''
def train_main_network(x):
	content_encoder.zero_grad()
	pose_encoder.zero_grad()
	decoder.zero_grad()

	# in accordance with the input specifications
	x_c1 = x[0]
	x_c2 = x[1]
	x_p1 = x[2]
	x_p2 = x[3]

	h_c1, skip1 = content_encoder(x_c1)
	h_c2, skip2 = content_encoder(x_c2)
	h_c2 = h_c2.detach()
	h_p1 = pose_encoder(x_p1)[0]
	h_p2 = pose_encoder(x_p2)[0].detach()

	# similarity loss: ||h_c1 - h_c2||
	sim_loss = mse_loss(h_c1, h_c2)

	# reconstruction loss: ||D(h_c1, h_p1), x_p1|| 
	rec = decoder(h_c1, skip1, h_p1)
	rec_loss = mse_loss(rec, x_p1)

	# scene discriminator loss
	target = torch.cuda.FloatTensor(args.batch_size, 1).fill_(0.5)
	out = scene_discriminator(h_p1, h_p2)
	sd_loss = bce_loss(out, Variable(target))

	# full loss
	loss = sim_loss + args.alpha*rec_loss + args.beta*sd_loss

	# backprop and update the prameters
	loss.backward()

	optimizer_content_encoder.step()
	optimizer_pose_encoder.step()
	optimizer_decoder.step()

	# return the similarity and reconstruction loss
	return sim_loss.data/args.batch_size, rec_loss.data/args.batch_size

'''
train the model
'''
def model_train(train_generator):
	content_encoder.train()
	pose_encoder.train()
	decoder.train()
	scene_discriminator.train()

	for iteration in range(args.max_iter):

		x = next(train_generator) # get the next input batch

		# train scene discriminator
		sd_loss, sd_acc = train_scene_discriminator(x)

		# train the main network
		sim_loss, rec_loss = train_main_network(x)

		iteration += 1

		# display the values
		print ('iteration: %d, sim loss: %0.8f, rec loss: %0.8f, sd loss: %0.8f, sd acc: %0.8f' 
				%(iteration, sim_loss, rec_loss, sd_loss, sd_acc))

		# save the values in an external file which can be plotted later
		with open(os.path.join('saved_values', 'loss_and_acc_%s.txt'%(args.dataset)), mode='a') as f:
			f.write('%0.8f %0.8f %0.8f %0.8f\n'%(sim_loss, rec_loss, sd_loss, sd_acc))

		# save the model weights for each of the components
		if iteration % args.save_iters == 0:
			save_dir = os.path.join(args.save_dir, 'iter_%d'%(iteration))
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			content_encoder.save_state_dict(os.path.join(save_dir,'content_encoder.pth'))
			pose_encoder.save_state_dict(os.path.join(save_dir,'pose_encoder.pth'))
			decoder.save_state_dict(os.path.join(save_dir,'decoder.pth'))
			scene_discriminator.save_state_dict(os.path.join(save_dir,'scene_discriminator.pth'))

'''
training the LSTM network
'''
def train_lstm(x):
	lstm.zero_grad()

	# initialise the hidden state and long term memory of LSTM with zero tensor
	lstm.h, lstm.c = lstm.init_hidden()

	# compute the fixed content feature from the last frame
	h_c, skip = content_encoder(x[args.n_past-1])
	h_c = h_c.detach()

	# compute the pose features for each of the time step
	h_p = [pose_encoder(x[i])[0].detach() for i in range(args.n_past+args.n_future)]

	mse = 0.0
	for i in range(1, args.n_past+args.n_future):
		pred = lstm(torch.cat([h_c, h_p[i-1]], dim=1)) # predict the pose features sequentially using the LSTM
		mse += mse_loss(pred, h_p[i]) # compare the predicted pose with the ground truth computed in line 238 and get the mse loss

	# backprop and update the parameters
	mse.backward()
	optimizer_lstm.step()

	# return the mse value after taking mean across all time steps
	return mse/(args.n_past+args.n_future)

'''
training the model consisting of the LSTM network (other network components are kept fixed)
'''
def model_lstm_train(train_generator):
	lstm.train()

	for iteration in range(args.max_iter):

		x = next(train_generator) # get the next training batch

		mse_loss = train_lstm(x) # get the mse loss

		iteration += 1

		# display the values
		print ('iteration: %d, mse loss: %0.8f'%(iteration, mse_loss))

		# save the values in an external file to be plotted later
		with open(os.path.join('saved_values', 'loss_and_acc_lstm_%s.txt'%(args.dataset)), mode='a') as f:
			f.write('%0.8f\n'%(mse_loss))

		# save the model weights
		if iteration % args.save_iters == 0:
			save_dir = os.path.join(args.save_dir, 'iter_%d'%(iteration))
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			lstm.save_state_dict(os.path.join(save_dir,'lstm.pth'))
 
if __name__ == '__main__':

	# get the train and test generator from the data laoder
	train_generator, test_generator = loader(args)

	# is start_iter is not zero, then load the model from the given iteration
	if args.start_iter != 0:
		content_encoder.load_state_dict(os.path.join(args.save_dir, 'iter_%d'%(args.start_iter), 'content_encoder.pth'), strict=False)
		pose_encoder.load_state_dict(os.path.join(args.save_dir, 'iter_%d'%(args.start_iter), 'pose_encoder.pth'), strict=False)
		decoder.load_state_dict(os.path.join(args.save_dir, 'iter_%d'%(args.start_iter), 'decoder.pth'), strict=False)
		scene_discriminator.load_state_dict(os.path.join(args.save_dir, 'iter_%d'%(args.start_iter), 'scene_discriminator.pth'), strict=False)
		lstm.load_state_dict(os.path.join(args.save_dir, 'iter_%d'%(args.start_iter), 'lstm.pth'), strict=False)

	if args.use_lstm:
		model_lstm_train(train_generator)
	else:
		model_train(train_generator)
