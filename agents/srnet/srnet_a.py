from __future__ import print_function
from __future__ import division
import tensorflow as tf

num_agent = 2
obs_dim = 27

send_out_dim = 3
recv_mid_dim = 5
recv_out_dim = 3

action_dim = 5

h_s_1 = h_s_2 = h_s_3 = 64
h_r_1 = h_r_2 = h_r_3 = 64
h_a_1 = h_a_2 = h_a_3 = 64


def generate_srnet(obs, conn, trainable=True):

    obs_list = list()
    sender_list = list()
    actor_list = list()
    recv_list = list()

    # Make observation
    for i in range(num_agent):
        obs_list.append(obs[:, i * obs_dim:(i + 1) * obs_dim])

    # Sender
    for i in range(num_agent):
        sender = generate_sender(obs_list[i])
        sender_list.append(sender)

    # Receiver
    for i in range(num_agent):
        with tf.variable_scope("recv"+str(i)):
            recv = generate_receiver(i, sender_list, conn)
            recv_list.append(recv)

    # Actor
    for i in range(num_agent):
        actor = generate_actor_softmax(obs_list[i], recv_list[i])
        actor_list.append(actor)

    actions = tf.concat(actor_list, axis=-1)
    return actions


def generate_receiver(a_id, sender_list, conn):
    """
    Make receiver network.
    This should be flexible to the varying number of agent within comm. range

    :param a_id: agent ID
    :param sender_list: receiving message
    :param conn: connectivity matrix (format: conn[recv][sender])
    :return:
    """

    recv = tf.zeros([1, recv_mid_dim], dtype=tf.float32, name=None)

    for i, msg in enumerate(sender_list):
        if i == a_id:
            continue
        r_filter = generate_recv_filter(msg)
        recv = tf.cond(conn[a_id][i], lambda: tf.add(r_filter, recv), lambda: recv)  # kdw

    return recv


def generate_recv_filter(msg):
    hidden_1 = tf.layers.dense(msg, h_r_1, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True, reuse=tf.AUTO_REUSE, name='dense_1')
    hidden_2 = tf.layers.dense(hidden_1, h_r_2, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True, reuse=tf.AUTO_REUSE, name='dense_2')
    hidden_3 = tf.layers.dense(hidden_2, h_r_3, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True, reuse=tf.AUTO_REUSE, name='dense_3')

    recv_filter = tf.layers.dense(hidden_3, recv_mid_dim, trainable=True, reuse=tf.AUTO_REUSE, name='dense_4')

    return recv_filter


def generate_sender(obs):
    hidden_1 = tf.layers.dense(obs, h_s_1, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True)
    hidden_2 = tf.layers.dense(hidden_1, h_s_2, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True)

    hidden_3 = tf.layers.dense(hidden_2, h_s_3, activation=tf.nn.relu,
                               kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                               bias_initializer=tf.constant_initializer(0.1),  # biases
                               use_bias=True, trainable=True)

    msg = tf.layers.dense(hidden_3, send_out_dim, trainable=True)

    return msg


def generate_actor(obs, r):
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

    action = tf.layers.dense(hidden_3, action_dim, trainable=True)

    return action


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


if __name__ == '__main__':
    obs = tf.placeholder(dtype=tf.float32, shape=[None, obs_dim * num_agent])
    conn = tf.placeholder(tf.bool, name='check')

    srnet = generate_srnet(obs, conn)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # conn[recv][sender]
    conn_i = [[True, True, True], [True, True, True], [True, True, True]]
    conn_i = [[False, True, False], [False, True, False], [False, True, False]]
    a, m, r = sess.run(srnet, feed_dict={obs: [[1, 2, 3, 4, 5, 6, 7, 8, 9]], conn: conn_i})

    print("Actions:", a)
    print("Message:", m)
    print("Recv.  :", r)

    exit()

    y = tf.placeholder(dtype=tf.float32, shape=[None, 3])

    cost = tf.reduce_sum(tf.square(y - srnet[0]))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    for i in range(100):
        sess.run(train, feed_dict={obs: [[1, 2, 3, 4, 5, 6, 7, 8, 9]], conn: conn_i, y: [[0.5, 0.5, 0.5]]})

    conn_i = [[False, True, False], [False, False, False], [False, True, False]]
    a, m, r = sess.run(srnet, feed_dict={obs: [[1, 2, 3, 4, 5, 6, 7, 8, 9]], conn: conn_i})

    print("Actions:", a)
    print("Message:", m)
    print("Recv.  :", r)
