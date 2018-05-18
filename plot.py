import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='mnist', help='which dataset values to check')
parser.add_argument('--use_lstm', type=bool, default=False, help='whether to check the lstm model')

args = parser.parse_args()

'''
Computes moving average of a numpy array

Arguments:
	a : numpy array
	n : moving window width

Returns:
	moving average with the given window size
'''
def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

'''
plots loss values stored in the given path
'''
def plot_loss(path):

	with open(path) as f:
		content = [x.strip('\n') for x in f.readlines()]
		content = [x.split(' ') for x in content]
		content = np.array(content)
		content = content[:1000,:]
	
	sim_loss = content[:,0].astype(float)
	rec_loss = content[:,1].astype(float)
	sd_loss = content[:,2].astype(float)
	sd_acc = content[:,3].astype(float)

	sim_loss = moving_average(sim_loss, 100)
	rec_loss = moving_average(rec_loss, 100)
	sd_loss = moving_average(sd_loss, 100)
	sd_acc = moving_average(sd_acc, 100)

	plt.plot(sim_loss)
	plt.show()
	plt.plot(rec_loss)
	plt.show()
	plt.plot(sd_loss)
	plt.show()
	plt.plot(sd_acc)
	plt.show()

'''
plots loss value of the lstm model stored in the specified path
'''
def plot_loss_with_lstm(path):

	with open(path) as f:
		content = [x.strip('\n') for x in f.readlines()]
		content = [x.split(' ') for x in content]
		content = np.array(content)

	mse_loss = content[:,0].astype(float)

	mse_loss = moving_average(mse_loss, 500)

	plt.plot(mse_loss)
	plt.show()

if __name__ == '__main__':
	if args.use_lstm:
		path = os.path.join('saved_values', 'loss_and_acc_lstm_%s.txt'%(args.dataset))
		plot_loss_with_lstm(path)
	else:
		path = os.path.join('saved_values', 'loss_and_acc_%s.txt'%(args.dataset))
		plot_loss(path)