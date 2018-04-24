#!/usr/bin/env python
# coding=utf8

"""
===========================================
 :mod:`qlearn` Q-Learning
===========================================
.. moduleauthor:: Daewoo Kim
.. note:: note...

설명
=====

Choose action based on q-learning algorithm
"""

import numpy as np
import tensorflow as tf
from agents.dqn_single_onehot.dq_network import *
from agents.dqn_single_onehot.replay_buffer import *
import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')


class Agent(object):

    def __init__(self, action_dim, obs_dim, name=""):
        logger.info("Q-Learning Agent")

        self._n_predator = FLAGS.n_predator
        self._n_prey = FLAGS.n_prey

        self._action_dim = action_dim
        self._obs_dim = obs_dim
        self._name = name
        self.update_cnt = 0
        self.target_update_period = 100

        self.df = FLAGS.df
        self.lr = FLAGS.lr

        # Make Q-network
        tf.reset_default_graph()
        my_graph = tf.Graph()

        with my_graph.as_default():
            self.sess = tf.Session(graph=my_graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
            self.q_network = DQNetwork(self.sess, self._obs_dim, self._action_dim)
            self.sess.run(tf.global_variables_initializer())

        self.replay_buffer = ReplayBuffer()

    def act(self, state):
        """
        For single agent, we set 'halt' action (2) for agent 2
        :param state:
        :return:
        """
        state_i = self.state_to_index(state)

        s = np.reshape(state_i, self._obs_dim)
        q = self.q_network.get_q_values(s[None])[0]

        action = np.random.choice(np.where(q == np.max(q))[0])

        return action, 2

    def train(self, state, action, reward, state_n, done):

        a = self.onehot(action[0], self._action_dim)
        s = self.state_to_index(state)
        s_n = self.state_to_index(state_n)
        r = np.sum(reward)

        self.store_sample(s, a, r, s_n, done)
        self.update_network()

        return 0

    def store_sample(self, s, a, r, s_n, done):

        self.replay_buffer.add_to_memory((s, a, r, s_n, done))
        return 0

    def update_network(self):
        self.update_cnt += 1
        if len(self.replay_buffer.replay_memory) < 10*minibatch_size:
            return 0

        minibatch = self.replay_buffer.sample_from_memory()
        self.q_network.training_qnet(minibatch)

        if self.update_cnt % self.target_update_period == 0:
            self.q_network.training_target_qnet()

        return 0

    def state_to_index(self, state):
        """
        For the single agent case, the state is only related to the position of agent 1
        :param state:
        :return:
        """
        p1, p2 = self.get_predator_pos(state)
        ret = self.onehot(p1, self._obs_dim)
        return ret

    def get_predator_pos(self, state):
        """
        return position of agent 1 and 2
        :param state: input is state
        :return:
        """
        state_list = list(np.array(state).ravel())
        return state_list.index(1), state_list.index(2)

    def onehot(self, index, size):
        n_hot = np.zeros(size)
        n_hot[index] = 1.0
        return n_hot
