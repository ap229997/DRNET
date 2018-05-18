import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
LSTM module used for sequential generation of pose features
which when concatenated with the content features and passed
through the decoder, gives the video frame at each stage
'''
class LSTM(nn.Module):
	'''
	Arguments:
		input_size : size of input to the lstm (concatenation of content and pose features)
		hidden_size : size of the hidden layer
		output_size : size of the output of the lstm (pose features dimensions)
		batch_size : batch size used
		layers : how many lstm layers to be used
	'''
	def __init__(self, input_size, hidden_size, output_size, batch_size, layers=1):
		super(LSTM, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.batch_size = batch_size
		self.layers = layers

		self.embedding = nn.Linear(input_size, hidden_size)
		# define an LSTM cell at each layer of the LSTM network
		# this cell when combined across each time step results in
		# sequential generation of video frames
		self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.layers)])
		self.fc = nn.Sequential(
					nn.Linear(hidden_size, output_size),
					nn.Tanh(),
					)

		# initialise the hidden state and the long term memory of the LSTM
		self.h, self.c = self.init_hidden()

	'''
	Initialise the hidden state and long term memory of the LSTM

	Arguments: None
	
	Returns:
		h, c : zero initialised hidden state and long term memory
	'''
	def init_hidden(self):
		# initialise h and c values with zero tensors
		h = []
		c = []
		for i in range(self.layers):
			h.append(Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda())
			c.append(Variable(torch.zeros(self.batch_size, self.hidden_size)).cuda())

		return h, c

	'''
	Computes a forward pass through the LSTM network at each time step

	Arguments:
		inp : input to the LSTM (generally the concatenation of content and pose features)

	Returns:
		out : output of the LSTM (generally the pose features)
	'''
	def forward(self, inp):
		inp = inp.view(-1, self.input_size) # input is reshaped to be in accordance with the defined LSTM specifications
		embedding = self.embedding(inp)

		for i in range(self.layers):
			self.h[i], self.c[i] = self.lstm[i](embedding, (self.h[i], self.c[i]))
			embedding = self.h[i]

		out = self.fc(embedding)

		return out

# testing
if __name__ == '__main__':
	torch.cuda.set_device(0)
	net = LSTM(128+5,256,5,2).cuda()
	inp = Variable(torch.FloatTensor(3,128+5)).cuda()
	out = net(inp)
	print ('out: ', out.shape)
