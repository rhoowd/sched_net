from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import sys
import random
from agents.schedule.replay_buffer import ReplayBuffer
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


class SchedulingAgent(object):

    def __init__(self, n_agent, action_dim, state_dim, obs_dim, name=""):
        logger.info("Schedule")

        self._n_agent = n_agent
        self._state_dim = state_dim
        self._action_dim_per_unit = action_dim
        self._obs_dim_per_unit = obs_dim

        # joint action space for critic network
        self._joint_action_dim = self._action_dim_per_unit ** self._n_agent
        # concatenated observation space
        self._obs_dim = self._obs_dim_per_unit * self._n_agent

        self._name = name
        self.update_cnt = 0

        # Make Actor Critic
        tf.reset_default_graph()
        my_graph = tf.Graph()

        with my_graph.as_default():
            self.sess = tf.Session(graph=my_graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
            self._actor = ActorNetwork(self.sess, self._n_agent, self._obs_dim, self._action_dim_per_unit, self._name)
            self._critic = CriticNetwork(self.sess, self._state_dim, self._joint_action_dim, self._name)

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

    def compose_joint_action(self, action_list):
        # compose action list into joint action
        r = 0
        for a in action_list:
            r = a + r * self._action_dim_per_unit
        return r

    def act(self, obs_list, schedule_list):

        # TODO just argmax when testing..

        action_prob_list = self._actor.action_for_state(np.concatenate(obs_list).reshape(1, self._obs_dim),
                                                        schedule_list.reshape(1, self._n_agent))
        if np.isnan(action_prob_list).any():
            raise ValueError('action_prob contains NaN')
        action_list = []
        for action_prob in action_prob_list.reshape(self._n_agent, self._action_dim_per_unit):
            action_list.append(np.random.choice(len(action_prob), p=action_prob))

        return action_list

    def schedule(self, obs_list):

        # pick one agent to communicate
        # TODO generalize for the number of senders

        schedule_prob = self._scheduler.schedule_for_obs(np.concatenate(obs_list)
                                                           .reshape(1, self._obs_dim))
        schedule_idx = np.random.choice(self._n_agent, p=schedule_prob[0])
        ret = np.zeros(self._n_agent)
        ret[schedule_idx] = 1.0
        return ret

    def train(self, state, obs_list, action_list, reward_list, state_next, obs_next_list, schedule_list, done):

        s = state
        o = obs_list
        a = action_list
        r = np.sum(reward_list)
        s_ = state_next
        o_ = obs_next_list
        c = schedule_list

        self.store_sample(s, o, a, r, s_, o_, c, done)
        self.update_ac()

    def store_sample(self, s, o, a, r, s_, o_, c, done):

        self.replay_buffer.add_to_memory((s, o, a, r, s_, o_, c, done))

    def update_ac(self):

        if len(self.replay_buffer.replay_memory) < 10 * FLAGS.m_size:
            return 0

        minibatch = self.replay_buffer.sample_from_memory()

        s, o, a, r, s_, o_, c, d = map(np.array, zip(*minibatch))

        a_concat = a
        a_joint = np.apply_along_axis(self.compose_joint_action, 1, a)
        o = np.reshape(o, [-1, self._obs_dim])
        o_ = np.reshape(o_, [-1, self._obs_dim])

        if FLAGS.schedule == 'schedule':
            td_error, _ = self._critic.training_critic(s, a_joint, r, s_, a_joint, d)  # train critic
            _ = self._actor.training_actor(o, a_concat, td_error, c)  # train actor
            _ = self._scheduler.training_scheduler(o, c, td_error)
            _ = self._critic.training_target_critic()  # train slow target critic

        else:
            td_error, _ = self._critic.training_critic(s, a_joint, r, s_, a_joint, d)  # train critic
            _ = self._actor.training_actor(o, a_concat, td_error, c)  # train actor
            _ = self._critic.training_target_critic()  # train slow target critic
