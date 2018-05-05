#!/usr/bin/env python
# coding=utf8

"""
===================================================================================
 :mod:`cfao` Critich with full state and Actor with limitied obs,  Separate Action
===================================================================================
.. moduleauthor:: Daewoo Kim
.. note:: note...

=====

"""

import numpy as np
import tensorflow as tf
import sys
from agents.schedule.replay_buffer import *
from agents.schedule.ac_network import ActorNetwork
from agents.schedule.ac_network import CriticNetwork
from agents.schedule.ac_network import SchedulerNetwork
from agents.evaluation import Evaluation
from agents.schedule import schedule_net

import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')


class Agent(object):

    def __init__(self, action_dim, obs_dim, name=""):
        logger.info("Schedule")

        self._n_predator = FLAGS.n_predator
        self._n_prey = FLAGS.n_prey
        self.map_size = FLAGS.map_size

        self._obs_dim = obs_dim
        self._action_dim_single = action_dim
        self._action_dim = action_dim * self._n_predator
        self._state_dim = (self.map_size**2) * (self._n_predator + self._n_prey)
        self._state_dim_single = (self.map_size**2)

        self._name = name
        self.update_cnt = 0

        # Make Actor Critic
        tf.reset_default_graph()
        my_graph = tf.Graph()

        self.input_dim = 3 * self._obs_dim * self._n_predator

        with my_graph.as_default():
            self.sess = tf.Session(graph=my_graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
            self._actor = ActorNetwork(self.sess, self.input_dim, self._action_dim, self._name)
            self._critic = CriticNetwork(self.sess, self._state_dim, self._action_dim, self._name)

            self._scheduler = SchedulerNetwork(self.sess, 3*self._obs_dim, self._n_predator)

            self.sess.run(tf.global_variables_initializer())

        self.replay_buffer = ReplayBuffer()

        self._eval = Evaluation()
        self.q_prev = None

    def act(self, state, obs, schedule):

        obs_i = self.obs_to_onehot(obs)
        o = np.reshape(obs_i, self.input_dim)
        q = self._actor.action_for_state(o[None], schedule[None])

        if np.isnan(q).any():
            print "Value Error: nan"
            print q
            sys.exit()

        a1 = np.random.choice(self._action_dim_single, p=q[0][0:5])
        a2 = np.random.choice(self._action_dim_single, p=q[0][5:10])

        return a1, a2

    def schedule(self, obs):
        obs_i = self.obs_to_onehot(obs)
        o = np.reshape(obs_i, self.input_dim)
        s_prob = self._scheduler.schedule_for_obs(o[None])
        s_i = np.random.choice(self._n_predator, p=s_prob[0])
        ret = np.zeros(self._n_predator)
        ret[s_i] = 1.0
        return ret

    def train(self, state, obs, action, reward, state_n, obs_n, done, schedule):

        a = self.action_to_index(action[0], action[1])
        a_h = self.action_to_nhot(a, 10)
        s = self.state_to_index(state)
        s_n = self.state_to_index(state_n)
        r = np.sum(reward)
        o = self.obs_to_onehot(obs)
        c = schedule
        c_i = self.schedule_to_onehot(schedule)

        self.store_sample(s, a, r, s_n, done, o, a_h, c, c_i)
        self.update_ac()

        return 0

    def store_sample(self, s, a, r, s_n, done, o, a_h, c, c_i):

        self.replay_buffer.add_to_memory((s, a, r, s_n, done, o, a_h, c, c_i))
        return 0

    def update_ac(self):
        if FLAGS.qtrace:
            self.update_cnt += 1
            if self.update_cnt % 2500 == 0:
                self.q()

        if len(self.replay_buffer.replay_memory) < 10 * FLAGS.m_size:
            return 0

        minibatch = self.replay_buffer.sample_from_memory()

        s = np.asarray([elem[0] for elem in minibatch])
        a = np.asarray([elem[1] for elem in minibatch])
        r = np.asarray([elem[2] for elem in minibatch])
        s_ = np.asarray([elem[3] for elem in minibatch])
        d = np.asarray([elem[4] for elem in minibatch])
        o = np.asarray([elem[5] for elem in minibatch])
        a_h = np.asarray([elem[6] for elem in minibatch])
        c = np.asarray([elem[7] for elem in minibatch])

        if FLAGS.schedule == 'schedule':
            td_error, _ = self._critic.training_critic(s, a, r, s_, a, d)  # train critic
            _ = self._actor.training_actor(o, a_h, td_error, c)  # train actor
            _ = self._scheduler.training_scheduler(o, c, td_error)
            _ = self._critic.training_target_critic()  # train slow target critic

        else:
            td_error, _ = self._critic.training_critic(s, a, r, s_, a, d)  # train critic
            _ = self._actor.training_actor(o, a_h, td_error, c)  # train actor
            _ = self._critic.training_target_critic()  # train slow target critic



        return 0

    def schedule_to_onehot(self, schedule):

        ret = np.zeros(self._n_predator)
        for i in range(self._n_predator):
            if schedule[i]:
                ret[i] = 1.0

        return ret

    def obs_to_onehot(self, obs):
        ret = list()

        for a in range(self._n_predator):
            wall = np.zeros(self._obs_dim)
            predator = np.zeros(self._obs_dim)
            prey = np.zeros(self._obs_dim)

            for i in range(self._obs_dim):
                if obs[a][i] == 1:
                    wall[i] = 1
                elif obs[a][i] == 3:
                    predator[i] = 1
                elif obs[a][i] == 4:
                    prey[i] = 1

            ret.append(np.concatenate([wall, predator, prey]))

        ret = np.concatenate(ret)
        return ret

    def state_to_index(self, state):
        """
        For the single agent case, the state is only related to the position of agent 1
        :param state:
        :return:
        """
        # p1, p2 = self.get_predator_pos(state)
        p1 = self.get_pos_by_id(state, 1)
        p2 = self.get_pos_by_id(state, 2)
        prey = self.get_pos_by_id(state, 3)

        ret = np.zeros(self._state_dim)
        ret[p1] = 1.0
        ret[p2 + self._state_dim_single] = 1.0
        ret[prey + 2*self._state_dim_single] = 1.0

        return ret

    def get_predator_pos(self, state):
        """
        return position of agent 1 and 2
        :param state: input is state
        :return:
        """
        state_list = list(np.array(state).ravel())
        return state_list.index(1), state_list.index(2)

    def get_pos_by_id(self, state, id):
        state_list = list(np.array(state).ravel())
        return state_list.index(id)

    def onehot(self, index, size):
        n_hot = np.zeros(size)
        n_hot[index] = 1.0
        return n_hot

    def index_to_action(self, index):
        return index % 5, index / 5

    def action_to_index(self, a1, a2):
        return a1 + a2 * 5

    def action_to_nhot(self, a_i, size):
        a1, a2 = self.index_to_action(a_i)
        ret = np.zeros(size)

        ret[a1] = 1.0
        ret[a2+5] = 1.0

        return ret

    def q(self):
        q_a = 0
        q_value = []
        for p1 in range(self.map_size ** 2):
            for p2 in range(self.map_size ** 2):
                if p1 == p2:
                    continue
                for prey in range(self.map_size ** 2):
                    if prey == p1 or prey == p2:
                        continue

                    s = np.zeros(self._state_dim)
                    s[p1] = 1.0
                    s[p2 + self._state_dim_single] = 1.0
                    s[prey + 2*self._state_dim_single] = 1.0

                    for a in range(25):
                        q = self._critic.get_critic_q(s[None], [a])
                        q_value.append(q)
                        q_a += q[0][0][0]

        print self.update_cnt, q_a/len(q_value)
