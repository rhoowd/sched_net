from agents.ind_ac.ac_network import ActorNetwork
from agents.ind_ac.ac_network import CriticNetwork
import numpy as np
import config

FLAGS = config.flags.FLAGS

class Agent(object):
    def __init__(self, action_dim, obs_dim, sess, name=""):
        self._action_dim = action_dim
        self._obs_dim = obs_dim
        self._name = name 
        self._actor = ActorNetwork(sess, self._obs_dim, self._action_dim, self._name)
        self._critic = CriticNetwork(sess, self._obs_dim, self._action_dim, self._name)

    def act(self, obs):
        return self._actor.action_for_state([obs])

    def train(self, minibatch, step):
        td_error, _ = self._critic.training_critic(np.asarray([elem[0] for elem in minibatch]),
                                             np.asarray([elem[2] for elem in minibatch]),
                                             np.asarray([elem[3] for elem in minibatch]),
                                             np.asarray([elem[4] for elem in minibatch]))

        _ = self._actor.training_actor(np.asarray([elem[0] for elem in minibatch]),
                                 np.asarray([elem[1] for elem in minibatch]),
                                 td_error)

        _ = self._critic.training_target_critic()
