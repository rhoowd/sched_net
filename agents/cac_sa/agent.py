#!/usr/bin/env python
# coding=utf8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import random
import numpy as np
import tensorflow as tf
import sys
from agents.cac_sa.replay_buffer import ReplayBuffer
from agents.cac_sa.ac_network import ActorNetwork
from agents.cac_sa.ac_network import CriticNetwork
from agents.evaluation import Evaluation

import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')


# centralized actor-critic agent guiding multiple predators with full observation
class SAPredatorAgentFO(object):

    def __init__(self, n_agent, action_dim, state_dim, obs_dim, name=""):
        logger.info("Centralized Actor-Critic")

        self._n_agent = n_agent
        self._state_dim = state_dim
        self._action_dim_per_unit = action_dim
        self._obs_dim_per_unit = obs_dim

        # concatenated action space for actor network
        self._concat_action_dim = self._action_dim_per_unit * self._n_agent
        self._joint_action_dim = self._action_dim_per_unit ** self._n_agent

        # fully-observable environment -> obs of all agents are identical
        self._obs_dim = self._obs_dim_per_unit * self._n_agent

        self._name = name
        self.update_cnt = 0

        # Make Actor Critic
        tf.reset_default_graph()
        my_graph = tf.Graph()

        with my_graph.as_default():
            self.sess = tf.Session(graph=my_graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

            self._actor = ActorNetwork(self.sess, self._n_agent, self._state_dim, self._action_dim_per_unit, self._name)
            self._critic = CriticNetwork(self.sess, self._n_agent, self._state_dim, self._action_dim_per_unit, self._name)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            if FLAGS.load_nn:
                if FLAGS.nn_file == "":
                    logger.error("No file for loading Neural Network parameter")
                    exit()
                self.saver.restore(self.sess, FLAGS.nn_file)

        self.replay_buffer = ReplayBuffer()

        self._eval = Evaluation()
        self.q_prev = None

    def save_nn(self, global_step):
        self.saver.save(self.sess, config.nn_filename, global_step)

    def explore(self):
        return [random.randrange(self._action_dim_per_unit)
                for _ in range(self._n_agent)]

    def decompose_joint_action(self, joint_action_index):
        # decompose joint action index into list of actions of each agent
        action_list = []
        for _ in range(self._n_agent):
            action_list.append(joint_action_index % self._action_dim_per_unit)
            joint_action_index //= self._action_dim_per_unit
        return action_list

    def compose_joint_action(self, action_list):
        # compose action list into joint action
        r = 0
        for a in action_list:
            r = a + r * self._action_dim_per_unit
        return r

    def act(self, state):

        # TODO just argmax when testing..
        # use obs_list in partially observable environment

        action_prob_list = self._actor.action_for_state(state.reshape(1, self._state_dim))

        if np.isnan(action_prob_list).any():
            raise ValueError('action_prob contains NaN')

        action_list = []
        for action_prob in action_prob_list.reshape(self._n_agent, self._action_dim_per_unit):
            action_list.append(np.random.choice(len(action_prob), p=action_prob))

        return action_list

    def train(self, state, obs_list, action_list, reward_list, state_next, done):

        # use obs_list in partially observable environment

        s = state
        a = action_list
        r = np.sum(reward_list)
        s_ = state_next

        self.store_sample(s, a, r, s_, done)
        self.update_ac()
        return 0

    def store_sample(self, s, a, r, s_, done):

        self.replay_buffer.add_to_memory((s, a, r, s_, done))
        return 0

    def update_ac(self):
        
        if len(self.replay_buffer.replay_memory) < 10 * FLAGS.m_size:
            return 0

        minibatch = self.replay_buffer.sample_from_memory()
        s, a, r, s_, d = map(np.array, zip(*minibatch))

        td_error, _ = self._critic.training_critic(s, a, r, s_, a, d)  # train critic
        _ = self._actor.training_actor(s, a, td_error)  # train actor

        _ = self._critic.training_target_critic()  # train slow target critic

        return 0