from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import config

FLAGS = config.flags.FLAGS


def generate_comm_network(obs_list, action_dim, n_agent, trainable=True, share=False, schedule=None):
    actions = list()
    h_num = 32

    if FLAGS.comm in [0, 1]:
        scope = "comm"
        for i in range(n_agent):
            if not share:
                scope = "comm" + str(i)

            with tf.variable_scope(scope):

                if FLAGS.comm == 0:  # Disconnected
                    agent_actor = disconnected_actor(obs_list[i], action_dim, h_num, trainable)
                elif FLAGS.comm == 1:  # Share obs of agent 0
                    agent_actor = share_one_obs(obs_list[i], obs_list[0], action_dim, h_num, trainable)
                else:
                    print("Wrong comm type")
                    exit()

            actions.append(agent_actor)

    elif FLAGS.comm in [2]:  # Full connection
        # Generate encoder
        encoder_scope = "encoder"
        capacity = FLAGS.capa
        encoder_list = list()
        for i in range(n_agent):
            if not FLAGS.e_share:
                encoder_scope = "encoder" + str(i)

            with tf.variable_scope(encoder_scope):
                encoder = encoder_network(obs_list[i], capacity, 32, 1)
            encoder_list.append(encoder)
        encoders = tf.concat(encoder_list, axis=1)
        # Generate actor
        scope = "comm"
        for i in range(n_agent):
            if not FLAGS.s_share:
                scope = "comm" + str(i)

            with tf.variable_scope(scope):
                agent_actor = comm_encoded_obs(obs_list[i], encoders, action_dim, h_num, trainable)

            actions.append(agent_actor)

    elif FLAGS.comm in [3]:  # Limited Connection (fixed k agent can communicate)
        # Generate encoder
        encoder_scope = "encoder"
        capacity = FLAGS.capa
        encoder_list = list()
        n_schedule_agent = FLAGS.s_num
        for i in range(n_schedule_agent):
            if not FLAGS.e_share:
                encoder_scope = "encoder" + str(i)

            with tf.variable_scope(encoder_scope):
                encoder = encoder_network(obs_list[i], capacity, 32, 1)
            encoder_list.append(encoder)
        encoders = tf.concat(encoder_list, axis=1)

        # Generate actor
        scope = "comm"
        for i in range(n_agent):
            if not FLAGS.s_share:
                scope = "comm" + str(i)

            with tf.variable_scope(scope):
                agent_actor = comm_encoded_obs(obs_list[i], encoders, action_dim, h_num, trainable)

            actions.append(agent_actor)

    elif FLAGS.comm in [4]:  # Scheduling (scheduled k agent can communicate)
        # Generate encoder
        encoder_scope = "encoder"
        aggr_scope = "aggr"
        capacity = FLAGS.capa
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
                    aggr_out = decode_aggregate_network(encoder_list, schedule, decoder_out_dim)
                aggr_list.append(aggr_out)

        else:
            with tf.variable_scope(aggr_scope):
                aggr_out = decode_aggregate_network(encoder_list, schedule, decoder_out_dim)
            for i in range(n_agent):
                aggr_list.append(aggr_out)

        # Generate actor
        scope = "comm"
        for i in range(n_agent):
            if not FLAGS.s_share:
                scope = "comm" + str(i)

            with tf.variable_scope(scope):
                # agent_actor = comm_encoded_obs(obs_list[i], encoders, action_dim, h_num, trainable)
                agent_actor = comm_encoded_obs(obs_list[i], aggr_list[i], action_dim, h_num, trainable)

            actions.append(agent_actor)

    return tf.concat(actions, axis=-1)


def disconnected_actor(obs, action_dim, h_num, trainable=True):

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


def share_one_obs(obs, comm, action_dim, h_num, trainable=True):
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
def encoder_network(e_input, out_dim, h_num, h_level, name="encoder", trainable=True):

    hidden = e_input
    for i in range(h_level):

        hidden = tf.layers.dense(hidden, h_num, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                                 bias_initializer=tf.constant_initializer(0.1),  # biases
                                 use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name=name+str(i))

    a = tf.layers.dense(hidden, out_dim, activation=tf.nn.relu,
                        kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                        bias_initializer=tf.constant_initializer(0.1),  # biases
                        use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name=name+"_out")
    return a


# Decoding and aggregating
def decode_aggregate_network(m_input_list, schedule, out_dim):

    aggregated_out = None

    for i, msg in enumerate(m_input_list):
        decoded_out = decode_network(msg, out_dim, 16, 1)
        schedule_i = tf.reshape(schedule[:, i], [tf.shape(schedule)[0], 1])
        schedule_extend = tf.add(schedule_i, tf.zeros([tf.shape(schedule)[0], out_dim]))
        scheduled_out = tf.multiply(decoded_out, schedule_extend)

        if aggregated_out is None:
            aggregated_out = scheduled_out
        else:
            aggregated_out = tf.add(aggregated_out, scheduled_out)

    return aggregated_out


def decode_network(m_input, out_dim, h_num, h_level, name="decoder", trainable=True):

    hidden = m_input
    for i in range(h_level):

        hidden = tf.layers.dense(hidden, h_num, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                                 bias_initializer=tf.constant_initializer(0.1),  # biases
                                 use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name=name+str(i))

    a = tf.layers.dense(hidden, out_dim, activation=tf.nn.relu,
                        kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                        bias_initializer=tf.constant_initializer(0.1),  # biases
                        use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name=name+"_out")
    return a
