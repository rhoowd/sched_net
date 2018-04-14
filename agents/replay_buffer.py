from collections import deque
import random
import config
import numpy as np

FLAGS = config.flags.FLAGS

replay_memory_capacity = FLAGS.replay_buffer_capacity  # capacity of experience replay memory
minibatch_size = FLAGS.minibatch_size  # size of minibatch from experience replay memory for updates
trace_length = FLAGS.rnn_trace_len

class ReplayBuffer:
	def __init__(self):
		self.replay_memory = deque(maxlen=replay_memory_capacity)

	def add_to_memory(self, experience):
		self.replay_memory.append(experience)

	def sample_from_memory(self):
		return random.sample(self.replay_memory, minibatch_size)

class RNNReplayBuffer:
	def __init__(self):
		self.replay_memory = deque(maxlen=replay_memory_capacity)
		self.paddings = None

	def add_to_memory(self, experience):
		self.replay_memory.append(experience)

		if self.paddings == None:
			obs = np.zeros(experience[-1][0].shape)
			self.paddings = (obs, 0, 0, obs, True)

	def pad_trace(self, trace):
		trace.extend([self.paddings]*(trace_length-len(trace)))
		return trace

	def sample_from_memory(self):
		if len(self.replay_memory) < minibatch_size:
			n_points_per_ep = int(np.ceil(minibatch_size * 1./len(self.replay_memory)))
			sampled_episodes = self.replay_memory
		else:
			n_points_per_ep = 1
			sampled_episodes = random.sample(self.replay_memory, minibatch_size)
		
		sampledTraces = []
		true_trace_length = np.ones(minibatch_size)*trace_length

		for i in range(n_points_per_ep):
			for j, episode in enumerate(sampled_episodes):
				if len(episode) < trace_length:					
					true_trace_length[j] = len(episode)
					sampledTraces.append(self.pad_trace(episode)) # use the whole episode
				else:
					point = np.random.randint(0,len(episode) + 1 - trace_length)
					sampledTraces.append(episode[point:point + trace_length])

		sampledTraces = np.array(sampledTraces[:minibatch_size]) # discard extra samples
		sampledTraces = np.reshape(sampledTraces,[minibatch_size*trace_length,-1])
		return sampledTraces, true_trace_length
