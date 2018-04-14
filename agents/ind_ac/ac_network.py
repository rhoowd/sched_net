#!/usr/bin/env python
# coding=utf8

import numpy as np
import tensorflow as tf
from collections import deque
import random

gamma = 0.9  # reward discount factor

h1_actor = 64  # hidden layer 1 size for the actor
h2_actor = 64  # hidden layer 2 size for the actor

h1_critic = 64  # hidden layer 1 size for the critic
h2_critic = 64  # hidden layer 2 size for the critic

# lr_actor = 0.0001  # learning rate for the actor
lr_actor = 1e-6  # learning rate for the actor
# lr_critic = 0.001  # learning rate for the critic
lr_critic = 1e-5  # learning rate for the critic
lr_decay = 1  # learning rate decay (per episode)

tau = 1e-2  # soft target update rate

np.set_printoptions(threshold=np.nan)

class ActorNetwork:
    def __init__(self, sess, state_dim, action_dim, nn_id=None):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        if nn_id == None:
            scope = 'actor'
        else:
            scope = 'actor_' + str(nn_id)

        # placeholders
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.action_ph = tf.placeholder(dtype=tf.int32, shape=[None])
        self.a_onehot = tf.one_hot(self.action_ph, self.action_dim, 1.0, 0.0)
        self.td_errors = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # indicators (go into target computation)
        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())  # for dropout

        # actor network
        with tf.variable_scope(scope):
            # Policy's outputted action for each state_ph (for generating actions and training the critic)
            self.actions = self.generate_actor_network(self.state_ph, trainable = True)

        # actor loss function (mean Q-values under current policy with regularization)
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        self.responsible = tf.multiply(self.actions, self.a_onehot)
        self.loss = tf.reduce_sum(tf.multiply(
                        tf.log(tf.reduce_sum(self.responsible,
                                            reduction_indices=1, keep_dims=True)), -self.td_errors)) 

        var_grads = tf.gradients(self.loss, self.actor_vars)

        self.actor_train_op = tf.train.AdamOptimizer(lr_actor * lr_decay).apply_gradients(zip(var_grads,self.actor_vars))


    # will use this to initialize both the actor network its slowly-changing target network with same structure
    def generate_actor_network(self, s, trainable):
        hidden = tf.layers.dense(s, h1_actor, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                 bias_initializer=tf.constant_initializer(0.1),  # biases
                                 use_bias=True, trainable=trainable, name='dense_a1')

        hidden_2 = tf.layers.dense(s, h2_actor, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                 bias_initializer=tf.constant_initializer(0.1),  # biases
                                 use_bias=True, trainable=trainable, name='dense_a2')

        actions_unscaled = tf.layers.dense(hidden_2, self.action_dim, activation=tf.nn.softmax,
                                           kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                           bias_initializer=tf.constant_initializer(0.1),  # biases
                                           trainable=trainable, name='dense_a3',
                                           use_bias=True)

        actions = actions_unscaled
        return actions

    def action_for_state(self, state_ph):
        return self.sess.run(self.actions,
                             feed_dict={self.state_ph: state_ph, self.is_training_ph: False})

    def training_actor(self, state_ph, action_ph, td_errors):
        return self.sess.run(self.actor_train_op,
                               feed_dict={self.state_ph: state_ph,
                                          self.action_ph: action_ph,
                                          self.td_errors: td_errors,
                                          self.is_training_ph: True})


class CriticNetwork:
    def __init__(self, sess, state_dim, action_dim, nn_id=None):

        self.sess = sess
        self.state_dim = state_dim

        if nn_id == None:
            scope = 'critic'
        else:
            scope = 'critic_' + str(nn_id)

        # placeholders
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None])  # indicators (go into target computation)
        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())  # for dropout

        with tf.variable_scope(scope):
            # Critic applied to state_ph
            self.q_values = self.generate_critic_network(self.state_ph, trainable=True)

        # slow target critic network
        with tf.variable_scope('slow_target_'+scope):
            self.slow_q_values = tf.stop_gradient(
                self.generate_critic_network(self.next_state_ph, trainable=False))

        # One step TD targets y_i for (s,a) from experience replay
        # = r_i + gamma*Q_slow(s',mu_slow(s')) if s' is not terminal
        # = r_i if s' terminal
        targets = tf.expand_dims(self.reward_ph, 1) + tf.expand_dims(self.is_not_terminal_ph, 1) * gamma * self.slow_q_values

        # 1-step temporal difference errors
        self.td_errors = targets - self.q_values

        critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        critic_loss = tf.reduce_mean(tf.square(self.td_errors))

        # critic optimizer
        self.critic_train_op = tf.train.AdamOptimizer(lr_critic * lr_decay).minimize(critic_loss, var_list=critic_vars)

        slow_target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_critic')
        update_slow_target_ops_c = []
        for i, slow_target_var in enumerate(slow_target_critic_vars):
            update_slow_target_critic_op = slow_target_var.assign(tau * critic_vars[i] + (1 - tau) * slow_target_var)
            # update_slow_target_critic_op = slow_target_var.assign(critic_vars[i]) #copy only
            update_slow_target_ops_c.append(update_slow_target_critic_op)
        self.update_slow_targets_op_c = tf.group(*update_slow_target_ops_c)

    # will use this to initialize both the critic network its slowly-changing target network with same structure
    def generate_critic_network(self, s, trainable):
        hidden = tf.layers.dense(s, h1_critic, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                 bias_initializer=tf.constant_initializer(0.1),  # biases
                                 use_bias=True, trainable=trainable, name='dense_c1')

        hidden_2 = tf.layers.dense(s, h2_critic, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                 bias_initializer=tf.constant_initializer(0.1),  # biases
                                 use_bias=True, trainable=trainable, name='dense_c2')

        q_values = tf.layers.dense(hidden_2, 1, trainable=trainable,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.1),  # biases
                                   name='dense_c3',use_bias=False)
        return q_values

    def training_critic(self, state_ph, reward_ph, next_state_ph, is_not_terminal_ph):

        return self.sess.run([self.td_errors, self.critic_train_op],
                             feed_dict={self.state_ph: state_ph,
                                        self.reward_ph: reward_ph,
                                        self.next_state_ph: next_state_ph,
                                        self.is_not_terminal_ph: is_not_terminal_ph,
                                        self.is_training_ph: True})

    def training_target_critic(self):
        return self.sess.run(self.update_slow_targets_op_c,
                             feed_dict={self.is_training_ph: False})