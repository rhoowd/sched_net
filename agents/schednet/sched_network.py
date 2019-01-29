#!/usr/bin/env python
# coding=utf8

import numpy as np
import tensorflow as tf
import config
from agents.schednet import comm

FLAGS = config.flags.FLAGS

h1_scheduler = 32  # hidden layer 1 size 
h2_scheduler = 32  # hidden layer 2 size 
lr_wg = FLAGS.w_lr   # learning rate for the weight generator
lr_decay = 1  # learning rate decay (per episode)
tau = 5e-2  # soft target update rate


class WeightGeneratorNetwork:
    def __init__(self, sess, n_player, obs_dim):

        self.sess = sess
        self.obs_dim = obs_dim # concatenated observation space
        self.n_player = n_player

        # placeholders
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.td_errors = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())  # for dropout

        self.sched_grads_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.n_player])

        with tf.variable_scope('schedule'):
            self.schedule_policy = self.generate_wg(self.obs_ph, trainable=True)

        with tf.variable_scope('slow_target_schedule'):
            self.target_schedule_policy = tf.stop_gradient(self.generate_wg(self.obs_ph, trainable=False))

        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='schedule')
        var_grads = tf.gradients(self.schedule_policy, self.actor_vars, -self.sched_grads_ph)
        self.scheduler_train_op = tf.train.AdamOptimizer(lr_wg * lr_decay).apply_gradients(
            zip(var_grads, self.actor_vars))

        slow_target_sch_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_schedule')

        # update values for slowly-changing targets towards current actor and critic
        update_slow_target_ops_i = []
        for i, slow_target_sch_var in enumerate(slow_target_sch_vars):
            update_slow_target_sch_op = slow_target_sch_var.assign(
                tau * self.actor_vars[i] + (1 - tau) * slow_target_sch_var)
            update_slow_target_ops_i.append(update_slow_target_sch_op)
        self.update_slow_targets_op_i = tf.group(*update_slow_target_ops_i)

    def generate_wg(self, obs, trainable=True):
        obs_list = list()
        sched_list = list()

        # [n_agents,n_samples,obs_dim]
        for i in range(self.n_player):
            obs_list.append(obs[:, i * self.obs_dim:(i + 1) * self.obs_dim]) 

        for i in range(self.n_player):
            s = self.generate_wg_network(obs_list[i], trainable)
            sched_list.append(s)

        schedule = tf.concat(sched_list, axis=-1)

        return schedule

    def generate_wg_network(self, obs, trainable=True):
        hidden_1 = tf.layers.dense(obs, h1_scheduler, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),
                                   bias_initializer=tf.constant_initializer(0.1),  
                                   use_bias=True, trainable=trainable)

        hidden_2 = tf.layers.dense(hidden_1, h2_scheduler, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),
                                   bias_initializer=tf.constant_initializer(0.1),  
                                   use_bias=True, trainable=trainable)

        schedule = tf.layers.dense(hidden_2, 1, activation=tf.nn.sigmoid, trainable=trainable)

        return schedule

    def schedule_for_obs(self, obs_ph):

        return self.sess.run(self.schedule_policy,
                             feed_dict={self.obs_ph: obs_ph, self.is_training_ph: False})[0]

    def target_schedule_for_obs(self, obs_ph):

        return self.sess.run(self.target_schedule_policy,
                             feed_dict={self.obs_ph: obs_ph, self.is_training_ph: False})

    def training_weight_generator(self, obs_ph, sched_grads_ph):

        return self.sess.run(self.scheduler_train_op,
                             feed_dict={self.obs_ph: obs_ph,
                                        self.sched_grads_ph: sched_grads_ph,
                                        self.is_training_ph: True})

    def training_target_weight_generator(self):
        return self.sess.run(self.update_slow_targets_op_i)
