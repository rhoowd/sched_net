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
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
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
        self._state_space = (FLAGS.map_size ** 2) ** self._n_predator
        self._action_space = self._action_dim ** self._n_predator
        self.q_table = np.zeros([self._state_space, self._action_space])
        logger.info("{}, {}".format(self._state_space, self._action_space))

        self.df = FLAGS.df
        self.lr = FLAGS.lr
        # self.num_step = 3000
        # self.step_cnt = 0
        # self.result_list = []

    def learn(self, train=True):

        print(self.index_to_action(1))
        print(self.action_to_index(self.index_to_action(1)))
        logger.debug('Start running (train: {})'.format(train))

        episode_cnt = 0
        success_cnt = 0

        while True:
            step_in_episode = 0
            obs_n = self._env.get_obs()  # init obs
            episode_cnt += 1
            while True:
                self.step_cnt += 1
                step_in_episode += 1

                action_n = self.act_n(obs_n, self.step_cnt)
                # action_n = self.act_n(obs_n, episode_cnt)
                prev_obs = obs_n
                obs_n, reward_n, done_n, info_n = self._env.step(action_n)
                # print("obs and reward", obs_n, reward_n, done_n)

                self.update_q_table(prev_obs[0], action_n[0], reward_n[0], obs_n[0])

                if reward_n[0] == 1:
                    success_cnt += 1


                if self.step_cnt % 100 == 0:
                    success_rate, step_per_episode = self.test()
                    print("Step:", self.step_cnt, "\tEpisode:", episode_cnt, "\tSuccess rate: ", success_rate, "\tStep per episode:", step_per_episode)
                    result.info("qlearn step "+str(self.step_cnt)+" eps "+str(episode_cnt)+" success_rate "+str(success_rate)+" step_per_episod "+str(step_per_episode))

                if done_n[0]:
                    self._env.reset()
                    break


            if self.step_cnt > self.num_step:
                break

        print("Q-table:")
        print(self.q_table)
        self.print_optimal_action()

    def test(self):

        episode_num = 100
        success_cnt = 0
        step = 0

        self._env.reset()

        for i in range(episode_num):
            obs_n = self._env.get_obs()  # init obs

            while True:
                step += 1

                action_n = self.act_n(obs_n, 0, train=False)
                obs_n, reward_n, done_n, info_n = self._env.step(action_n)

                if reward_n[0] == 1:
                    success_cnt += 1

                if done_n[0]:
                    self._env.reset()
                    break

                if step > 10000:
                    break

        success_rate = float(success_cnt)/episode_num
        step_per_episode = float(step)/episode_num
        return success_rate, step_per_episode

    def print_optimal_action(self):
        for i in range(self.num_discrete_obs):
            action = np.argmax(self.q_table[i,:] + np.random.randn(1, 4) / (3000+1))
            if i % self.num_obs_side == 0:
                print("")
            if action == 0:
                print('↓',)
            elif action == 1:
                print('↑',)
            elif action == 2:
                print('→',)
            elif action == 3:
                print('←',)
        print("")

    def act_n(self, obs_n, step, train=True):
        action_n = []
        for drone_id in range(self._n_drone):
            action_n.append(self.act(obs_n[drone_id], step, drone_id))

        return action_n

    def act(self, state):

        state_i = self.state_to_index(state)
        action_i = np.argmax(self.q_table[state_i, :])

        return self.index_to_action(action_i)

    def train(self, state, action, reward, state_n):

        # print("reward", reward, np.sum(reward))
        # print("state", state)
        # print("action", action)
        # print("state_n", state_n)
        # print("")

        a = self.action_to_index(action[0], action[1])
        s = self.state_to_index(state)
        s_n = self.state_to_index(state_n)
        r = np.sum(reward)
        self.q_table[s, a] = (1 - self.lr) * self.q_table[s, a] + self.lr * (r + self.df * np.max(self.q_table[s_n, :]))

        return 0

    def state_to_index(self, state):
        p1, p2 = self.get_predator_pos(state)
        ret = p1 * 9 + p2
        return ret

    def get_predator_pos(self, state):
        state_list = list(np.array(state).ravel())
        return state_list.index(1), state_list.index(2)

    def index_to_action(self, index):
        return index // 5, index % 5

    def action_to_index(self, a1, a2):
        return a1*5 + a2


