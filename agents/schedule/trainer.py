#!/usr/bin/env python
# coding=utf8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import six
import numpy as np
from agents.schedule.agent import SchedulingAgent
from agents.simple_agent import RandomAgent, StaticAgent
from agents.evaluation import Evaluation
import logging
import config
from envs.gui import canvas

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')

training_step = FLAGS.training_step
testing_step = FLAGS.testing_step

epsilon_dec = 1.0/training_step
epsilon_min = 0.01


class Trainer(object):

    def __init__(self, env):
        logger.info("Schedule Trainer is created")

        self._env = env
        self._eval = Evaluation()
        self._agent_profile = self._env.get_agent_profile()
        self._state_dim = self._env.get_info()[0]['state'].shape[0]
        self._n_predator = self._agent_profile['predator']['n_agent']

        # single scheduling agent
        self._predator_agent = SchedulingAgent(n_agent=self._n_predator,
                                               action_dim=self._agent_profile["predator"]["act_dim"],
                                               state_dim=self._state_dim,
                                               obs_dim=self._agent_profile["predator"]["obs_dim"][0])

        
        self._prey_agent = []
        if FLAGS.moving_prey:
            # randomly moving prey agent
            NewAgent = RandomAgent
            prey_param = 5
        else:
            NewAgent = StaticAgent
            prey_param = 2
        for _ in range(self._agent_profile['prey']['n_agent']):
            self._prey_agent.append(NewAgent(prey_param))

        self.epsilon = 0.3

        # For gui
        self.canvas = canvas.Canvas(self._n_predator, 1, FLAGS.map_size)
        self.canvas.setup()


    def learn(self):

        global_step = 0
        episode_num = 0
        print_flag = True

        while global_step < training_step:
            episode_num += 1
            step_in_ep = 0
            obs_n = self._env.reset()
            state = self._env.get_info()[0]['state']
            total_reward = 0

            while True:

                global_step += 1
                step_in_ep += 1

                predator_schedule = self.get_schedule(obs_n, global_step, FLAGS.schedule)

                action_n = self.get_action(obs_n, global_step, predator_schedule)

                obs_n_next, reward_n, done_n, info_n = self._env.step(action_n)
                state_next = info_n[0]['state']

                # print(state_next * FLAGS.map_size, predator_schedule)
                self.canvas.draw(state_next * FLAGS.map_size, predator_schedule, "Hello")

                done_single = sum(done_n) > 0
                self.train_agents(state, obs_n, action_n, reward_n, state_next, obs_n_next, predator_schedule, done_single)

                obs_n = obs_n_next
                state = state_next
                total_reward += np.sum(reward_n)

                if is_episode_done(done_n, global_step):
                    self.canvas.draw(state_next * FLAGS.map_size, predator_schedule, "Hello", True)
                    if print_flag:
                        print("[train_ep %d]" % (episode_num),"\tstep:", global_step, "\tep_step:", step_in_ep, "\treward", total_reward)
                    break

                if global_step % FLAGS.eval_step == 0:
                    self.test(global_step)
                    break

        self._predator_agent.save_nn(global_step)
        self._eval.summarize()

    def get_schedule(self, obs_n, global_step, type='schedule', train=True):

        predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]

        # TODO generalize for the number of senders
        if type == 'schedule':
            if train and (global_step < FLAGS.m_size * FLAGS.pre_train_step or np.random.rand() < self.epsilon):
                i = np.random.choice(range(self._n_predator), 1)
                ret = np.full(self._n_predator, 0.0)
                ret[i] = 1.0
                return ret
            else:
                ret = self._predator_agent.schedule(predator_obs)
        elif type == 'connect':
            ret = np.full(self._n_predator, 1.0)
        elif type == 'disconnect':
            ret = np.full(self._n_predator, 0.0)
        elif type == 'random':
            ret = np.random.choice([1.0, 0.0], self._n_predator)
        elif type == 'random_one':
            i = np.random.choice(range(self._n_predator), 1)
            ret = np.full(self._n_predator, 0.0)
            ret[i] = 1.0
        elif type == 'one':
            ret = np.full(self._n_predator, 0.0)
            ret[0] = 1.0
        else:
            ret = None

        return ret

    def get_action(self, obs_n, global_step, predator_schedule, train=True):
        act_n = [0] * len(obs_n)
        self.epsilon = max(self.epsilon - epsilon_dec, epsilon_min)

        # Action of predator
        if train and (global_step < FLAGS.m_size * FLAGS.pre_train_step or np.random.rand() < self.epsilon):  # with prob. epsilon
            predator_action = self._predator_agent.explore()
        else:
            predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
            predator_action = self._predator_agent.act(predator_obs, predator_schedule)

        for i, idx in enumerate(self._agent_profile['predator']['idx']):
            act_n[idx] = predator_action[i]

        # Action of prey
        for i, idx in enumerate(self._agent_profile['prey']['idx']):
            act_n[idx] = self._prey_agent[i].act(None)

        return np.array(act_n, dtype=np.int32)

    def train_agents(self, state, obs_n, action_n, reward_n, state_next, obs_n_next, predator_schedule, done):
        # train predator only
        predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
        predator_action = [action_n[i] for i in self._agent_profile['predator']['idx']]
        predator_reward = [reward_n[i] for i in self._agent_profile['predator']['idx']]
        predator_obs_next = [obs_n_next[i] for i in self._agent_profile['predator']['idx']] # predator_obs_next
        self._predator_agent.train(state, predator_obs, predator_action, predator_reward, state_next,
                                   predator_obs_next, predator_schedule, done)

    def test(self, curr_ep=None):

        global_step = 0
        episode_num = 0

        test_flag = FLAGS.kt

        while global_step < testing_step:
            episode_num += 1
            obs_n = self._env.reset()
            state = self._env.get_info()[0]['state']
            if test_flag:
                print("\nInit\n", state)
            total_reward = 0

            step_in_ep = 0

            while True:

                global_step += 1
                step_in_ep += 1

                predator_schedule = self.get_schedule(obs_n, global_step, FLAGS.schedule)

                action_n = self.get_action(obs_n, global_step, predator_schedule, False)
                obs_n_next, reward_n, done_n, info_n = self._env.step(action_n)
                state_next = info_n[0]['state']

                if test_flag:
                    aa = six.moves.input('>')
                    if aa == 'c':
                        test_flag = False
                    print(action_n)
                    print(state_next)

                obs_n = obs_n_next
                state = state_next
                total_reward += np.sum(reward_n)

                if is_episode_done(done_n, global_step, "test") or step_in_ep > FLAGS.max_step:
                    break

        print("Test result: Average steps to capture: ", curr_ep, float(global_step)/episode_num)
        self._eval.update_value("test_result", float(global_step)/episode_num, curr_ep)


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


