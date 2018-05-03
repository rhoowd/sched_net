#!/usr/bin/env python
# coding=utf8

"""
===========================================
 :mod:`cac` Centralized Actor-Critic
===========================================
.. moduleauthor:: Daewoo Kim
.. note:: note...

설명
map 3 일때
 CDQN - 10
 Random - 80
 CAC - 20
=====

Choose action based on q-learning algorithm
"""
from __future__ import print_function
from __future__ import division
import random
import numpy as np
import tensorflow as tf
import sys
from replay_buffer import ReplayBuffer
from agents.cac_fo_generalized.ac_network import ActorNetwork
from agents.cac_fo_generalized.ac_network import CriticNetwork
from agents.evaluation import Evaluation

import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')

# centralized actor-critic agent guiding multiple predators with full observation
class JointPredatorAgentFO(object):

    def __init__(self, n_agent, action_dim, obs_dim, name=""):
        logger.info("Centralized Actor-Critic")

        self._n_agent = n_agent
        self._action_dim_per_unit = action_dim
        self._obs_dim_per_unit = obs_dim

        # joint action space
        self._action_dim = self._action_dim_per_unit ** self._n_agent
        # fully-observable environment -> obs of all agents are identical
        self._obs_dim = self._obs_dim_per_unit

        self._name = name
        self.update_cnt = 0

        # Make Actor Critic
        tf.reset_default_graph()
        my_graph = tf.Graph()

        with my_graph.as_default():
            self.sess = tf.Session(graph=my_graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

            self._actor = ActorNetwork(self.sess, self._obs_dim, self._action_dim, self._name)
            self._critic = CriticNetwork(self.sess, self._obs_dim, self._action_dim, self._name)

            self.sess.run(tf.global_variables_initializer())

        self.replay_buffer = ReplayBuffer()

        self._eval = Evaluation()
        self.q_prev = None

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
        return reduce(lambda x, y: x * self._action_dim_per_unit + y,
                      reversed(action_list))

    def act(self, obs):

        # TODO just argmax when testing..
        # use obs_list in partially observable environment

        action_prob = self._actor.action_for_state(obs.reshape(1, self._obs_dim))

        if np.isnan(action_prob).any():
            print("Value Error: nan")
            print(action_prob)
            sys.exit()

        joint_action_index = np.random.choice(len(action_prob[0]), p=action_prob[0])

        return self.decompose_joint_action(joint_action_index)

    def train(self, obs, action_list, reward_list, 
              obs_next, done):

        # use obs_list in partially observable environment

        s = obs
        a = self.compose_joint_action(action_list)
        r = np.sum(reward_list)
        s_ = obs_next

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

        if FLAGS.use_action_in_critic:
            a_ = self._actor.target_action_for_next_state(s_).argmax(axis=1)  # get actions for next state
            td_error, _ = self._critic.training_critic(s, a, r, s_, a_, d)  # train critic
            _ = self._actor.training_actor(s, a, td_error)  # train actor
            _ = self._actor.training_target_actor()  # train slow target actor
        else:
            td_error, _ = self._critic.training_critic(s, a, r, s_, a, d)  # train critic
            _ = self._actor.training_actor(s, a, td_error)  # train actor

        _ = self._critic.training_target_critic()  # train slow target critic

        return 0

