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
import logging
import config
from agents.qlearn_c.agent import Agent
from agents.simple_agent import StaticAgent as NonLearningAgent

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')

training_step = FLAGS.training_step
testing_step = FLAGS.testing_step

epsilon_dec = 1.0/training_step
epsilon_min = 0.1


class Trainer(object):

    def __init__(self, env):
        logger.info("Q-Learning Trainer is created")

        self._env = env
        self._n_predator = FLAGS.n_predator
        self._n_prey = FLAGS.n_prey
        self._agent_profile = self._env.get_agent_profile()
        self._agent_precedence = self._env.agent_precedence

        self._agent = Agent(self._agent_profile["predator"]["act_dim"], self._agent_profile["predator"]["obs_dim"][0])
        self._prey_agent = NonLearningAgent(2)

        self.epsilon = 1.0

    def learn(self):

        step = 0
        episode = 0
        print_flag = False

        while step < training_step:
            episode += 1
            obs = self._env.reset()
            state = self._env.get_full_encoding()[:, :, 2]

            total_reward = 0

            if print_flag:
                print "[train_ep %d]" % (episode),
            ep_step = 0

            while True:
                step += 1
                ep_step += 1
                self.epsilon = max(self.epsilon - epsilon_dec, epsilon_min)

                action = self.get_action(obs, step, state)
                obs_n, reward, done, info = self._env.step(action)
                state_n = self._env.get_full_encoding()[:, :, 2]

                self.train_agents(state, action, reward, state_n)

                obs = obs_n
                state = state_n
                total_reward += np.sum(reward)

                if is_episode_done(done, step):
                    if print_flag:
                        print "\tstep:", step, "\tep_step:", ep_step, "\treward", total_reward
                    break

    def get_action(self, obs, step, state, train=True):
        act_n = []

        # Action of predator
        if train and np.random.rand() < self.epsilon:  # with prob. epsilon
            for i in range(self._n_predator):
                # guided action (no collision)
                shape = int(np.sqrt(obs[i].shape[0] / FLAGS.history_len))
                imap = obs[i].reshape((FLAGS.history_len, shape, shape))
                minimap = imap[-1, :, :]

                valid_act = [2]
                center = shape // 2
                if minimap[center - 1, center] == 0:  # up
                    valid_act.append(0)
                if minimap[center, center - 1] == 0:  # left
                    valid_act.append(1)
                if minimap[center, center + 1] == 0:  # right
                    valid_act.append(3)
                if minimap[center + 1, center] == 0:  # down
                    valid_act.append(4)
                action = np.random.choice(valid_act)
                act_n.append(action)
                continue
        else:
            action_list = self._agent.act(state)
            for a in action_list:
                act_n.append(a)

        # Action of prey
        act_n.append(self._prey_agent.act(None))

        return np.array(act_n, dtype=np.int32)

    def train_agents(self, state, action, reward, state_n):
        self._agent.train(state, action, reward, state_n)

    def test(self):

        print "\n== Test ==\n"
        step = 0
        episode = 0

        while step < testing_step:
            episode += 1
            obs = self._env.reset()
            state = self._env.get_full_encoding()[:, :, 2]

            total_reward = 0

            print "[test_ep  %d]" % (episode),
            ep_step = 0

            while True:
                step += 1
                ep_step += 1

                action = self.get_action(obs, step, state, False)
                obs_n, reward, done, info = self._env.step(action)
                state_n = self._env.get_full_encoding()[:, :, 2]

                obs = obs_n
                state = state_n
                total_reward += np.sum(reward)

                if is_episode_done(done, step):
                    print "\tstep:", step, "\tep_step:", ep_step, "\treward", total_reward
                    break

        print "\nTest result:\n Average steps to capture: ", float(step)/episode, "\n"



def is_episode_done(done, step):

    if sum(done) > 0 or step > FLAGS.training_step:
        return True
    else:
        return False


