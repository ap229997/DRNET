import torch
from torch.utils.data import DataLoader
import argparse
import utils

data_type = torch.cuda.FloatTensor

def get_training_batch(train_loader, args):
	while True:
		for sequence in train_loader:
			batch = utils.normalize_data(args, data_type, sequence)
			yield batch

def get_testing_batch(test_loader, args):
	while True:
		for sequence in test_loader:
			batch = utils.normalize_data(args, data_type, sequence)
			yield batch

'''
data loader for train and test data

Arguments:
	args : load the argument parser from the main file

Returns:
	train_generator : generator function for training data
	test_generator : generator function for testing data
'''
def loader(args):
	# get the train and test dataset
	train_data, test_data = utils.load_dataset(args)

	# use the pytorch dataloader for efficiently loading dataset
	train_loader = DataLoader(train_data, num_workers=args.data_threads, batch_size=args.batch_size,
								shuffle=True, drop_last=True, pin_memory=True)
	test_loader = DataLoader(test_data, num_workers=args.data_threads, batch_size=args.batch_size,
								shuffle=True, drop_last=True, pin_memory=True)

	# get the generator functions for the training and testing data
	train_generator = get_training_batch(train_loader, args)
	test_generator = get_testing_batch(test_loader, args)

	return train_generator, test_generator