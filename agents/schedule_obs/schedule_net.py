from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np

send_out_dim = 3
recv_out_dim = 5

action_dim = 5

h_num = 16

h_s_1 = h_s_2 = h_s_3 = h_num
h_r_1 = h_r_2 = h_r_3 = h_num
h_a_1 = h_a_2 = h_a_3 = h_num

# Flags
flag_sender_share = True
flag_receiver_share = False


def generate_schedulenet(obs, schedule, num_agent, obs_dim_agent, trainable=True):

    obs_list = list()
    sender_list = list()
    actor_list = list()
    recv_list = list()

    # Make observation
    for i in range(num_agent):
        obs_list.append(obs[:, i * obs_dim_agent:(i + 1) * obs_dim_agent])

    # Sender
    for i in range(num_agent):
        if flag_sender_share:
            sender = generate_sender(obs_list[i], trainable)
        else:
            with tf.variable_scope("sender" + str(i)):
                sender = generate_sender(obs_list[i], trainable)
        sender_list.append(sender)

    # Receiver
    for i in range(num_agent):
        if flag_receiver_share:
            recv = generate_receiver(i, sender_list, schedule)
        else:
            with tf.variable_scope("recv" + str(i)):
                recv = generate_receiver(i, sender_list, schedule)

        recv_list.append(recv)

    # Actor
    for i in range(num_agent):
        if flag_receiver_share:
            actor = generate_actor_softmax(obs_list[i], recv_list[0])
        else:
            actor = generate_actor_softmax(obs_list[i], recv_list[i])
        actor_list.append(actor)

    actions = tf.concat(actor_list, axis=-1)
    return actions


def generate_receiver(a_id, sender_list, schedule):
    """
    Make receiver network.
    This should be flexible to the varying number of agent within comm. range

    :param a_id: agent ID
    :param sender_list: receiving message
    :param schedule: schedule vectore (dim: recv_out_dim * num_agent)
    :return:
    """
    recv = tf.zeros([tf.shape(sender_list)[1], recv_out_dim], dtype=tf.float32, name=None)

    for i, msg in enumerate(sender_list):
        r_filter_out = generate_recv_filter(msg)
        s_i = schedule[:, i * recv_out_dim:(i + 1) * recv_out_dim]
        r_filter_out_with_schedule = tf.multiply(r_filter_out, s_i)
        recv = tf.add(recv, r_filter_out_with_schedule)

    return recv


def generate_recv_filter(msg):
    hidden_1 = tf.layers.dense(msg, h_r_1, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True, reuse=tf.AUTO_REUSE, name='r_dense_1')
    hidden_2 = tf.layers.dense(hidden_1, h_r_2, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True, reuse=tf.AUTO_REUSE, name='r_dense_2')
    hidden_3 = tf.layers.dense(hidden_2, h_r_3, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True, reuse=tf.AUTO_REUSE, name='r_dense_3')

    recv_filter = tf.layers.dense(hidden_3, recv_out_dim, trainable=True, reuse=tf.AUTO_REUSE, name='r_dense_4')

    return recv_filter


def generate_sender(obs, trainable=True):
    hidden_1 = tf.layers.dense(obs, h_s_1, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name='s_dense_1')
    hidden_2 = tf.layers.dense(hidden_1, h_s_2, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name='s_dense_2')

    hidden_3 = tf.layers.dense(hidden_2, h_s_3, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=trainable, reuse=tf.AUTO_REUSE, name='s_dense_3')

    msg = tf.layers.dense(hidden_3, send_out_dim, trainable=trainable, reuse=tf.AUTO_REUSE, name='s_dense_4')

    return msg


def generate_actor_softmax(obs, r):
    input = tf.concat([obs, r], axis=-1)

    hidden_1 = tf.layers.dense(input, h_a_1, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True)
    hidden_2 = tf.layers.dense(hidden_1, h_a_2, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True)

    hidden_3 = tf.layers.dense(hidden_2, h_a_3, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True)

    action = tf.layers.dense(hidden_3, action_dim, activation=tf.nn.softmax,
                             kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                             bias_initializer=tf.constant_initializer(0.1),  # biases
                             trainable=True,
                             use_bias=True)

    return action


def schedule_to_vector(schedule):
    ret = []
    for s_sample in schedule:
        ret_sample = []
        for s in s_sample:
            ret_sample = np.append(ret_sample, np.full(recv_out_dim, s))
        ret.append(ret_sample)

    ret = np.array(ret)
    return ret


if __name__ == '__main__':
    obs_dim = 5
    num_agent = 2

    obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim * num_agent])
    schedule_ph = tf.placeholder(dtype=tf.float32, shape=[None, recv_out_dim * num_agent])

    schedule_net = generate_schedulenet(obs_ph, schedule_ph, num_agent, obs_dim)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    obs_data = [range(10)]
    obs_data = [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]
    schedule_vector = schedule_to_vector([[1.0, 1.0]])
    print("schedule:", schedule_vector)

    result = sess.run(schedule_net, feed_dict={obs_ph: obs_data, schedule_ph: schedule_vector})
    print("result:", np.array(result))

    exit()

    y = tf.placeholder(dtype=tf.float32, shape=[None, 3])

    cost = tf.reduce_sum(tf.square(y - schedule_net[0]))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    schedule_vector = schedule_to_vector([[0.0, 0.0]])
    for i in range(100):
        sess.run(train, feed_dict={obs_ph: [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], schedule_ph: schedule_vector, y: [[0.5, 0.5, 0.5]]})

    obs_data = [range(10)]
    obs_data = [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]
    schedule_vector = schedule_to_vector([[1.0, 1.0]])
    print("schedule:", schedule_vector)

    result = sess.run(schedule_net, feed_dict={obs_ph: obs_data, schedule_ph: schedule_vector})
    print("result:", np.array(result))
