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
import math
from agents.cdqn_fixed.dq_network import *
from agents.cdqn_fixed.replay_buffer import *
from agents.evaluation import Evaluation

import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')


class Agent(object):

    def __init__(self, action_dim, obs_dim, name=""):
        logger.info("Centralized DQN Agent")


        self._n_predator = FLAGS.n_predator
        self._n_prey = FLAGS.n_prey
        self.map_size = FLAGS.map_size


        self._action_dim = action_dim ** self._n_predator
        self._obs_dim = obs_dim * self._n_predator

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

        self._eval = Evaluation()
        self.q_prev = None

    def act(self, state):
        """
        For single agent, we set 'halt' action (2) for agent 2
        :param state:
        :return:
        """
        state_i = self.state_to_index(state)

        s = np.reshape(state_i, self._obs_dim)
        q = self.q_network.get_q_values(s[None])[0]

        action_i = np.random.choice(np.where(q == np.max(q))[0])

        action = self.index_to_action(action_i)

        # return action[0], 2
        return action[0], action[1]

    def train(self, state, action, reward, state_n, done):

        a_i = self.action_to_index(action[0], action[1])
        a = self.onehot(a_i, self._action_dim)
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
        if len(self.replay_buffer.replay_memory) < FLAGS.pre_train_step*minibatch_size:
            return 0

        minibatch = self.replay_buffer.sample_from_memory()
        self.q_network.training_qnet(minibatch)

        if self.update_cnt % self.target_update_period == 0:
            self.q_network.training_target_qnet()
            if FLAGS.qtrace:
                if self.update_cnt % 1000 == 0:
                    self.q_diff()

        return 0

    def state_to_index(self, state):
        """
        For the single agent case, the state is only related to the position of agent 1
        :param state:
        :return:
        """
        p1, p2 = self.get_predator_pos(state)
        ret = np.zeros(self._obs_dim)
        ret[p1] = 1.0
        ret[p2] = 1.0
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

    def index_to_action(self, index):
        return index % 5, index / 5

    def action_to_index(self, a1, a2):
        return a1 + a2 * 5

    def q_diff(self):

        if self.q_prev == None:
            self.q_prev = self.q()
            return

        q_next = self.q()

        d = 0.0
        a = 0.0

        for i in range(self._obs_dim):
            for j in range(self._action_dim):
                d += math.fabs(self.q_prev[i][j] - q_next[i][j])
                a += q_next[i][j]
        avg = a/(self._obs_dim*self._action_dim)

        self._eval.update_value("q_avg", avg, self.update_cnt)
        self._eval.update_value("q_diff", d, self.update_cnt)

        self.q_prev = q_next

        print self.update_cnt, d, avg

    def q(self):
        q_value = []
        for p1 in range(self.map_size ** 2):
            if p1 == 0:
                continue
            for p2 in range(self.map_size ** 2):
                if p1 == p2 or p2 == 0:
                    continue
                s = np.zeros(self._obs_dim)
                s[p1] = 1.0
                s[p2] = 1.0

                q = self.q_network.get_target_q_values(s[None])[0]
                q_value.append(q)

        return q_value
