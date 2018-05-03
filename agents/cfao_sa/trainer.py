#!/usr/bin/env python
# coding=utf8

"""
============================================================================
 :mod:`cfao` Critic with Full state, Actor with limited obs separate action
============================================================================
.. moduleauthor:: Daewoo Kim
.. note:: note...

설명
=====

Choose action based on q-learning algorithm
python main.py --agent cdqn_fo --training_step 50000 --map_size 4 --scenario pursuit --lr 0.0001
"""
from __future__ import print_function
from __future__ import division
import numpy as np
from agents.cfao_sa.agent import Agent
from agents.simple_agent import RandomAgent as NonLearningAgent
from agents.evaluation import Evaluation
import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')

training_step = FLAGS.training_step
testing_step = FLAGS.testing_step

epsilon_dec = 1.0/training_step
epsilon_min = 0.1


class Trainer(object):

    def __init__(self, env):
        logger.info("Centralized DQN Trainer is created")

        self._env = env
        self._eval = Evaluation()
        self._n_predator = FLAGS.n_predator
        self._n_prey = FLAGS.n_prey
        self._agent_profile = self._env.get_agent_profile()
        self._agent_precedence = self._env.agent_precedence

        self._agent = Agent(self._agent_profile["predator"]["act_dim"], self._agent_profile["predator"]["obs_dim"][0])
        self._prey_agent = NonLearningAgent(5)

        self.epsilon = 0.3

    def learn(self):

        step = 0
        episode = 0
        print_flag = False

        while step < training_step:
            episode += 1
            ep_step = 0
            obs = self._env.reset()
            state = self._env.get_full_encoding()[:, :, 2]
            total_reward = 0

            while True:
                step += 1
                ep_step += 1

                action = self.get_action(obs, step, state)

                obs_n, reward, done, info = self._env.step(action)
                state_n = self._env.get_full_encoding()[:, :, 2]

                # print(obs_n)
                # imap = np.array(obs_n[0]).reshape((3, 3))
                # print(state_n)
                # print(np.array(obs_n[0]).reshape((3, 3)))
                # print(np.array(obs_n[1]).reshape((3, 3)))
                # print("")

                done_single = sum(done) > 0
                self.train_agents(state, obs, action, reward, state_n, obs_n, done_single)

                obs = obs_n
                state = state_n
                total_reward += np.sum(reward)

                if is_episode_done(done, step):
                    if print_flag:
                        print("[train_ep %d]" % (episode),"\tstep:", step, "\tep_step:", ep_step, "\treward", total_reward)
                    break

            if episode % FLAGS.eval_step == 0:
                self.test(episode)

        self._eval.summarize()

    def get_action(self, obs, step, state, train=True):
        act_n = []
        self.epsilon = max(self.epsilon - epsilon_dec, epsilon_min)

        # Action of predator
        if train and (step < FLAGS.m_size * FLAGS.pre_train_step or np.random.rand() < self.epsilon):  # with prob. epsilon
            for i in range(self._n_predator):  # guided action (no collision)
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
            action_list = self._agent.act(state, obs)
            for a in action_list:
                act_n.append(a)

        # Action of prey
        act_n.append(self._prey_agent.act(None))
        # act_n[1] = 2

        return np.array(act_n, dtype=np.int32)

    def train_agents(self, state, obs, action, reward, state_n, obs_n, done):
        self._agent.train(state, obs, action, reward, state_n, obs_n, done)

    def test(self, curr_ep=None):

        step = 0
        episode = 0

        test_flag = FLAGS.kt

        while step < testing_step:
            episode += 1
            obs = self._env.reset()
            state = self._env.get_full_encoding()[:, :, 2]
            if test_flag:
                print("\nInit\n", state)
            total_reward = 0

            ep_step = 0

            while True:

                step += 1
                ep_step += 1

                action = self.get_action(obs, step, state, False)
                obs_n, reward, done, info = self._env.step(action)
                state_n = self._env.get_full_encoding()[:, :, 2]

                if test_flag:
                    aa = raw_input('>')
                    if aa == 'c':
                        test_flag = False
                    print(action)
                    print(state_n)

                obs = obs_n
                state = state_n
                total_reward += np.sum(reward)

                if is_episode_done(done, step, "test") or ep_step > FLAGS.max_step:
                    break

        print("Test result: Average steps to capture: ", curr_ep, float(step)/episode)
        self._eval.update_value("test_result", float(step)/episode, curr_ep)


def is_episode_done(done, step, e_type="train"):

    if e_type == "test":
        if sum(done) > 0 or step >= FLAGS.testing_step:
            return True
        else:
            return False

    else:
        if sum(done) > 0 or step >= FLAGS.training_step:
            return True
        else:
            return False


