#!/usr/bin/env python
# coding=utf8

import numpy as np
import tensorflow as tf
import config
from agents.comm_obs import comm

FLAGS = config.flags.FLAGS

h1_scheduler = 32  # hidden layer 1 size for the critic
h2_scheduler = 32  # hidden layer 2 size for the critic

lr_actor = FLAGS.a_lr   # learning rate for the actor
lr_critic = FLAGS.c_lr  # learning rate for the critic
lr_decay = 1  # learning rate decay (per episode)

tau = 5e-2  # soft target update rate

np.set_printoptions(threshold=np.nan)

class SchedulerNetwork:
    def __init__(self, sess, n_player, obs_dim):

        self.sess = sess
        self.obs_dim = obs_dim # concatenated observation space
        self.n_player = n_player

        # placeholders
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.schedule_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.n_player])

        self.td_errors = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())  # for dropout

        # actor network
        with tf.variable_scope('schedule'):
            schedule_policy = self.generate_scheduler(self.obs_ph, trainable=True)
            self.schedule_policy = tf.nn.softmax(schedule_policy)

        # actor loss function (mean Q-values under current policy with regularization)
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='schedule')
        self.responsible = tf.multiply(self.schedule_policy, self.schedule_ph)  # =\pi (policy)

        log_prob = tf.log(tf.reduce_sum(self.responsible, reduction_indices=1, keep_dims=True))
        entropy = -tf.reduce_sum(self.schedule_policy * tf.log(self.schedule_policy), 1)

        self.loss = tf.reduce_sum(-(tf.multiply(log_prob, self.td_errors) + 0.01 * entropy))

        var_grads = tf.gradients(self.loss, self.actor_vars)
        self.grad = var_grads
        self.scheduler_train_op = tf.train.AdamOptimizer(lr_actor * lr_decay).apply_gradients(
            zip(var_grads, self.actor_vars))

    def generate_scheduler(self, obs, trainable=True):
        obs_list = list()
        sched_list = list()

        # [n_agents,n_samples,obs_dim]
        for i in range(self.n_player):
            obs_list.append(obs[:, i * self.obs_dim:(i + 1) * self.obs_dim]) 

        for i in range(self.n_player):
            s = self.generate_schedule_network(obs_list[i], trainable)
            sched_list.append(s)

        schedule = tf.concat(sched_list, axis=-1)

        # schedule_softmax = tf.nn.softmax(schedule)

        return schedule

    def generate_schedule_network(self, obs, trainable=True):
        hidden_1 = tf.layers.dense(obs, h1_scheduler, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                                   bias_initializer=tf.constant_initializer(0.1),  # biases
                                   use_bias=True, trainable=trainable)

        hidden_2 = tf.layers.dense(hidden_1, h2_scheduler, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                                   bias_initializer=tf.constant_initializer(0.1),  # biases
                                   use_bias=True, trainable=trainable)

        schedule = tf.layers.dense(hidden_2, 1, trainable=trainable)

        return schedule

    def schedule_for_obs(self, obs_ph):

        return self.sess.run(self.schedule_policy,
                             feed_dict={self.obs_ph: obs_ph, self.is_training_ph: False})[0]

    def training_scheduler(self, obs_ph, schedule_ph, td_errors):

        return self.sess.run(self.scheduler_train_op,
                             feed_dict={self.obs_ph: obs_ph,
                                        self.schedule_ph: schedule_ph,
                                        self.td_errors: td_errors,
                                        self.is_training_ph: True})
