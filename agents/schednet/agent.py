#!/usr/bin/env python
# coding=utf8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import random
import numpy as np
import tensorflow as tf

from agents.schednet.replay_buffer import ReplayBuffer
from agents.schednet.ac_network import ActorNetwork
from agents.schednet.ac_network import CriticNetwork
from agents.schednet.sched_network import SchedulerNetwork
from agents.evaluation import Evaluation

import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')


# centralized actor-critic agent guiding multiple predators with full observation
class PredatorAgent(object):

    def __init__(self, n_agent, action_dim, state_dim, obs_dim, name=""):
        logger.info("CCentralized Critic Independent Actor")

        self._n_agent = n_agent
        self._state_dim = state_dim + n_agent
        self._action_dim_per_unit = action_dim
        self._obs_dim_per_unit = obs_dim + 1

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
            self._actor = ActorNetwork(self.sess, self._n_agent, self._obs_dim_per_unit, self._action_dim_per_unit, self._name)
            self._critic = CriticNetwork(self.sess, self._n_agent, self._state_dim, self._name)
            self._scheduler = SchedulerNetwork(self.sess, self._n_agent, self._obs_dim)

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

    def act(self, obs_list, schedule_list):

        # TODO just argmax when testing..
        # use obs_list in partially observable environment

        action_prob_list = self._actor.action_for_state(np.concatenate(obs_list)
                                                          .reshape(1, self._obs_dim),
                                                        schedule_list.reshape(1, self._n_agent))

        if np.isnan(action_prob_list).any():
            raise ValueError('action_prob contains NaN')

        action_list = []
        for action_prob in action_prob_list.reshape(self._n_agent, self._action_dim_per_unit):
            action_list.append(np.random.choice(len(action_prob), p=action_prob))

        return action_list

    def train(self, state, obs_list, action_list, reward_list, state_next, obs_next_list, schedule_n, priority, done):

        # use obs_list in partially observable environment

        s = state
        o = obs_list
        a = action_list
        r = np.sum(reward_list)
        s_ = state_next
        o_ = obs_next_list
        c = schedule_n
        p = priority

        self.store_sample(s, o, a, r, s_, o_, c, p, done)
        self.update_ac()
        return 0

    def store_sample(self, s, o, a, r, s_, o_, c, p, done):

        self.replay_buffer.add_to_memory((s, o, a, r, s_, o_, c, p, done))
        return 0

    def update_ac(self):
        
        if len(self.replay_buffer.replay_memory) < FLAGS.pre_train_step * FLAGS.m_size:
            return 0

        minibatch = self.replay_buffer.sample_from_memory()
        s, o, a, r, s_, o_, c, p, d = map(np.array, zip(*minibatch))
        o = np.reshape(o, [-1, self._obs_dim])
        o_ = np.reshape(o_, [-1, self._obs_dim])

        p_ = self._scheduler.target_schedule_for_obs(o_)
        
        td_error, _ = self._critic.training_critic(s, r, s_, p, p_, d)  # train critic
        _ = self._actor.training_actor(o, a, c, td_error)  # train actor

        sch_grads = self._critic.grads_for_scheduler(s, p)
        _ = self._scheduler.training_scheduler(o, sch_grads)
        _ = self._critic.training_target_critic()  # train slow target critic
        _ = self._scheduler.training_target_scheduler()

        return 0


    def schedule(self, obs_list):
        priority = self._scheduler.schedule_for_obs(np.concatenate(obs_list)
                                                           .reshape(1, self._obs_dim))

        if FLAGS.sch_type == "top":
            schedule_idx = np.argsort(-priority)[:FLAGS.s_num]
        elif FLAGS.sch_type == "softmax":
            sm = softmax(priority)
            schedule_idx = np.random.choice(self._n_agent, p=sm)
        else: # IF N_SUM == 1
            schedule_idx = np.argmax(priority)
                            
        ret = np.zeros(self._n_agent)
        ret[schedule_idx] = 1.0
        return ret, priority

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
