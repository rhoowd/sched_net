from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np


def generate_comm_network(obs_list, action_dim, n_agent, trainable=True, share=False):
    actions = list()
    h_num = 32

    for i in range(n_agent):
        if share:
            # i_actor = self.generate_inpep_actor_network(obs_list[i], trainable)
            i_actor = generate_indep_actor_network(obs_list[i], action_dim, h_num, trainable)
        else:
            with tf.variable_scope("iactor" + str(i)):
                # i_actor = self.generate_inpep_actor_network(obs_list[i], trainable)
                i_actor = generate_indep_actor_network(obs_list[i], action_dim, h_num, trainable)
        actions.append(i_actor)

    return tf.concat(actions, axis=-1)


def generate_indep_actor_network(obs, action_dim, h_num, trainable=True):
    hidden_1 = tf.layers.dense(obs, h_num, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name='ia_dense_1')
    hidden_2 = tf.layers.dense(hidden_1, h_num, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name='ia_dense_2')

    hidden_3 = tf.layers.dense(hidden_2, h_num, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name='ia_dense_3')

    a = tf.layers.dense(hidden_3, action_dim, activation=tf.nn.softmax,
                        kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                        bias_initializer=tf.constant_initializer(0.1),  # biases
                        trainable=trainable, name='ia_dense_4',
                        use_bias=True)
    return a


def generate_comm_network_0_schedule(obs_list, action_dim, n_agent, trainable=True, share=False):
    actions = list()
    h_num = 32

    for i in range(n_agent):
        if share:
            i_actor = generate_0_schedule(obs_list[i], obs_list[0], action_dim, h_num, trainable)
        else:
            with tf.variable_scope("iactor" + str(i)):
                i_actor = generate_0_schedule(obs_list[i], obs_list[0], action_dim, h_num, trainable)
        actions.append(i_actor)

    return tf.concat(actions, axis=-1)


def generate_0_schedule(obs, comm, action_dim, h_num, trainable=True):
    c_input = tf.concat([obs, comm], axis=1)
    hidden_1 = tf.layers.dense(c_input, h_num, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name='ia_dense_1')
    hidden_2 = tf.layers.dense(hidden_1, h_num, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name='ia_dense_2')

    hidden_3 = tf.layers.dense(hidden_2, h_num, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name='ia_dense_3')

    a = tf.layers.dense(hidden_3, action_dim, activation=tf.nn.softmax,
                        kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                        bias_initializer=tf.constant_initializer(0.1),  # biases
                        trainable=trainable, name='ia_dense_4',
                        use_bias=True)
    return a
