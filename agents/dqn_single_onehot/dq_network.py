import numpy as np
import tensorflow as tf
from collections import deque
import random
import config

FLAGS = config.flags.FLAGS

gamma = FLAGS.df  # reward discount factor
learning_rate = FLAGS.lr
h1 = 32
h2 = 32
h3 = 32

replay_memory_capacity = 50000  # capacity of experience replay memory
minibatch_size = FLAGS.m_size  # size of minibatch from experience replay memory for updates

class DQNetwork(object):
    def __init__(self, sess, state_dim, action_dim):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        # placeholders
        self.s_in = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        self.a_in = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim])
        self.y_in = tf.placeholder(dtype=tf.float32, shape=[None])

        # indicators (go into target computation)
        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())  # for dropout

        with tf.variable_scope('q_network'):
            self.q_network = self.generate_dqn(self.s_in, True)

        with tf.variable_scope('target_q_network'):
            self.target_q_network = self.generate_dqn(self.s_in, False)

        with tf.variable_scope('optimization'):
            self.Q_act = tf.reduce_sum(self.q_network*self.a_in, reduction_indices=1)
            self.cost = tf.reduce_sum(tf.square(self.y_in - self.Q_act))
            self.train_network = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        global_vars = tf.GraphKeys.GLOBAL_VARIABLES
        t_params = tf.get_collection(global_vars, scope='target_q_network')
        e_params = tf.get_collection(global_vars, scope='q_network')
        self.update_target_fn = []
        for var, var_target in zip(sorted(e_params, key=lambda v: v.name), sorted(t_params, key=lambda v: v.name)):
            self.update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*self.update_target_fn)

    def generate_dqn(self, s, trainable=True):
        hidden_1 = tf.layers.dense(s, h1, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.1),  # biases
                                   use_bias=True,
                                   trainable=trainable, name='dense_h1')
        hidden_2 = tf.layers.dense(hidden_1, h2, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.1),  # biases
                                   use_bias=True,
                                   trainable=trainable, name='dense_h2')

        hidden_3 = tf.layers.dense(hidden_2, h3, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.1),  # biases
                                   use_bias=True,
                                   trainable=trainable, name='dense_h3')

        q_values = tf.layers.dense(hidden_3, self.action_dim, trainable=trainable)

        return q_values

    def get_q_values(self, state_ph):
        return self.sess.run(self.q_network, feed_dict={self.s_in: state_ph})

    def training_qnet(self, minibatch):
        y = []

        # Get target value from target network
        q_minibatch = self.sess.run(self.target_q_network, feed_dict={self.s_in: [data[3] for data in minibatch]})

        for i in range(minibatch_size):  # For all samples in minibatch
            if minibatch[i][4]:  # if terminal
                y.append(minibatch[i][2])
            else:
                max_q = np.max(q_minibatch[i])
                y.append(minibatch[i][2] + gamma * max_q)

        self.sess.run(self.train_network, feed_dict={
                self.y_in: y,
                self.a_in: [data[1] for data in minibatch],
                self.s_in: [data[0] for data in minibatch]
            })

    def training_target_qnet(self):
        """
        copy weights from q_network to target q_network
        :return:
        """
        self.sess.run(self.update_target_fn)


