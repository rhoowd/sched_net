# coding=utf8

from __future__ import print_function, division, absolute_import
import random
import numpy as np
import tensorflow as tf

from agents.schednet.replay_buffer import ReplayBuffer
from agents.schednet.ac_network import ActionSelectorNetwork
from agents.schednet.ac_network import CriticNetwork
from agents.schednet.sched_network import WeightGeneratorNetwork
from agents.evaluation import Evaluation

import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger('Agent')
result = logging.getLogger('Result')


class PredatorAgent(object):

    def __init__(self, n_agent, action_dim, state_dim, obs_dim, name=""):
        logger.info("Predator Agent is created")

        self._n_agent = n_agent
        self._state_dim = state_dim
        self._action_dim_per_unit = action_dim
        self._obs_dim_per_unit = obs_dim
        self._obs_dim = self._obs_dim_per_unit * self._n_agent

        self._name = name
        self.update_cnt = 0

        # Make Networks
        tf.reset_default_graph()
        my_graph = tf.Graph()

        with my_graph.as_default():
            self.sess = tf.Session(graph=my_graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

            self.action_selector = ActionSelectorNetwork(self.sess, self._n_agent, self._obs_dim_per_unit, self._action_dim_per_unit, self._name)
            self.weight_generator = WeightGeneratorNetwork(self.sess, self._n_agent, self._obs_dim)
            self.critic = CriticNetwork(self.sess, self._n_agent, self._state_dim, self._name)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            if FLAGS.load_nn:
                if FLAGS.nn_file == "":
                    logger.error("No file for loading Neural Network parameter")
                    exit()
                self.saver.restore(self.sess, FLAGS.nn_file)

        self.replay_buffer = ReplayBuffer()
        self._eval = Evaluation()

    def save_nn(self, global_step):
        self.saver.save(self.sess, config.nn_filename, global_step)

    def act(self, obs_list, schedule_list):

        action_prob_list = self.action_selector.action_for_state(np.concatenate(obs_list)
                                                                   .reshape(1, self._obs_dim),
                                                                 schedule_list.reshape(1, self._n_agent))

        if np.isnan(action_prob_list).any():
            raise ValueError('action_prob contains NaN')

        action_list = []
        for action_prob in action_prob_list.reshape(self._n_agent, self._action_dim_per_unit):
            action_list.append(np.random.choice(len(action_prob), p=action_prob))

        return action_list

    def train(self, state, obs_list, action_list, reward_list, state_next, obs_next_list, schedule_n, priority, done):

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

        p_ = self.weight_generator.target_schedule_for_obs(o_)
        
        td_error, _ = self.critic.training_critic(s, r, s_, p, p_, d)  # train critic
        _ = self.action_selector.training_actor(o, a, c, td_error)  # train actor

        wg_grads = self.critic.grads_for_scheduler(s, p)
        _ = self.weight_generator.training_weight_generator(o, wg_grads)
        _ = self.critic.training_target_critic()  # train slow target critic
        _ = self.weight_generator.training_target_weight_generator()

        return 0

    def schedule(self, obs_list):
        priority = self.weight_generator.schedule_for_obs(np.concatenate(obs_list)
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

    def explore(self):
        return [random.randrange(self._action_dim_per_unit)
                for _ in range(self._n_agent)]


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
