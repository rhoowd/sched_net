import numpy as np

class RandomAgent(object):
    def __init__(self, action_dim):
        self._action_dim = action_dim

    def act(self, obs):
        return np.random.randint(self._action_dim)

    def train(self, minibatch, step):
        return

class StaticAgent(object):
    def __init__(self, action):
        self._action = action

    def act(self, obs):
        return self._action

    def train(self, minibatch, step):
        return