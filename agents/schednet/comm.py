from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import config
import sys

FLAGS = config.flags.FLAGS


def generate_comm_network(obs_list, obs_dim_per_unit, action_dim, n_agent, trainable=True, share=False, schedule=None):
    actions = list()
    h_num = 32

    capacity = FLAGS.capa 

    # Generate encoder
    encoder_scope = "encoder"
    aggr_scope = "aggr"
    decoder_out_dim = 16
    encoder_list = list()

    for i in range(n_agent):
        if not FLAGS.e_share:
            encoder_scope = "encoder" + str(i)

        with tf.variable_scope(encoder_scope):
            encoder = encoder_network(obs_list[i], capacity, 32, 1)
        encoder_list.append(encoder)

    aggr_list = list()
    if not FLAGS.a_share:
        for i in range(n_agent):
            aggr_scope = "aggr" + str(i)
            with tf.variable_scope(aggr_scope):
                aggr_out = decode_concat_network(encoder_list, schedule, capacity, decoder_out_dim)
            aggr_list.append(aggr_out)

    else:
        with tf.variable_scope(aggr_scope):
            aggr_out = decode_concat_network(encoder_list, schedule, capacity, decoder_out_dim)
        for i in range(n_agent):
            aggr_list.append(aggr_out)

    # Generate actor
    scope = "comm"
    for i in range(n_agent):
        if not FLAGS.s_share:
            scope = "comm" + str(i)

        with tf.variable_scope(scope):
            agent_actor = comm_encoded_obs(obs_list[i], aggr_list[i], action_dim, h_num, trainable)

        actions.append(agent_actor)

    return tf.concat(actions, axis=-1)


def comm_encoded_obs(obs, c_input, action_dim, h_num, trainable=True):
    c_input = tf.concat([obs, c_input], axis=1)
    hidden_1 = tf.layers.dense(c_input, h_num, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name='sender_1')
    hidden_2 = tf.layers.dense(hidden_1, h_num, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name='sender_2')

    hidden_3 = tf.layers.dense(hidden_2, h_num, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name='sender_3')

    a = tf.layers.dense(hidden_3, action_dim, activation=tf.nn.softmax,
                        kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                        bias_initializer=tf.constant_initializer(0.1),  # biases
                        use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name='sender_4')
    return a


# Encoding
def encoder_network(e_input, out_dim, h_num, h_level, name="encoder", trainable=FLAGS.trainable_encoder):

    hidden = e_input
    for i in range(h_level):

        hidden = tf.layers.dense(hidden, h_num, activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                                    bias_initializer=tf.constant_initializer(0.1),  # biases
                                    use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name=name+str(i))

    ret = tf.layers.dense(hidden, out_dim, activation=tf.nn.relu,
                        kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                        bias_initializer=tf.constant_initializer(0.1),  # biases
                        use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name=name+"_out")
    return ret

def decode_concat_network(m_input_list, schedule, capacity, out_dim):

    inp = tf.stack(m_input_list, axis=-2)
    masked_msg = tf.boolean_mask(tf.reshape(inp, [-1, capacity]), tf.reshape(tf.cast(schedule, tf.bool), [-1]))
    return tf.reshape(masked_msg, [-1, FLAGS.s_num * capacity], name='scheduled')
