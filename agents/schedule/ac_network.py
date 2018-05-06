#!/usr/bin/env python
# coding=utf8
"""
Layer 2 -> 3
lr_actor = 1e-5  # learning rate for the actor
lr_critic = 1e-4  # learning rate for the critic
training step: 10000
df: 0.999
flags.DEFINE_integer("b_size", 10000, "Size of the replay memory")
flags.DEFINE_integer("m_size", 64, "Minibatch size")
flags.DEFINE_integer("pre_train_step", 10, "during [m_size * pre_step] take random action")
epsilon = 0.1
epsilon_min = 0.01

"""
import numpy as np
import tensorflow as tf
import config
from agents.schedule import schedule_net

FLAGS = config.flags.FLAGS

gamma = FLAGS.df  # reward discount factor

h_num = 64
h1_actor = h_num  # hidden layer 1 size for the actor
h2_actor = h_num  # hidden layer 2 size for the actor
h3_actor = h_num  # hidden layer 3 size for the actor

h1_critic = h_num  # hidden layer 1 size for the critic
h2_critic = h_num  # hidden layer 2 size for the critic
h3_critic = h_num  # hidden layer 3 size for the critic

h1_scheduler = h_num  # hidden layer 1 size for the critic
h2_scheduler = h_num  # hidden layer 2 size for the critic
h3_scheduler = h_num  # hidden layer 3 size for the critic


lr_critic = FLAGS.c_lr  # learning rate for the critic
lr_actor = FLAGS.a_lr  # learning rate for the actor
lr_schedule = FLAGS.s_lr  # learning rate for the critic

lr_decay = 1  # learning rate decay (per episode)
tau = FLAGS.s_lr   # soft target update rate

np.set_printoptions(threshold=np.nan)


class ActorNetwork:
    def __init__(self, sess, state_dim, action_dim, obs_dim_agent, nn_id=None):
        self.sess = sess
        self.state_dim = state_dim
        self.obs_dim_agent = obs_dim_agent
        self.action_dim = action_dim
        self.n_agent = FLAGS.n_predator

        if nn_id == None:
            scope = 'actor'
        else:
            scope = 'actor_' + str(nn_id)

        # placeholders
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        # self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim])
        self.schedule_ph = tf.placeholder(dtype=tf.float32, shape=[None, schedule_net.recv_out_dim * self.n_agent])
        self.td_errors = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # indicators (go into target computation)
        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())  # for dropout

        # actor network
        with tf.variable_scope(scope):
            # Policy's outputted action for each state_ph (for generating actions and training the critic)
            self.actions = self.generate_actor_network(self.state_ph, self.schedule_ph, self.n_agent, self.obs_dim_agent, trainable=True)

        # actor loss function (mean Q-values under current policy with regularization)
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        self.responsible = tf.multiply(self.actions, self.action_ph)

        log_prob = tf.log(1e-10+tf.reduce_sum(self.responsible, reduction_indices=1, keep_dims=True))
        entropy = -tf.reduce_sum(self.actions * tf.log(1e-10+self.actions), 1)
        # 1e-10 is added to avoid log(0) which causes nan value error

        self.loss = tf.reduce_sum(-(tf.multiply(log_prob, self.td_errors) + 0.01 * entropy))

        var_grads = tf.gradients(self.loss, self.actor_vars)
        self.actor_train_op = tf.train.AdamOptimizer(lr_actor * lr_decay).apply_gradients(
            zip(var_grads, self.actor_vars))

        slow_target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_' + scope)

        # update values for slowly-changing targets towards current actor and critic
        update_slow_target_ops_a = []
        for i, slow_target_actor_var in enumerate(slow_target_actor_vars):
            update_slow_target_actor_op = slow_target_actor_var.assign(
                tau * self.actor_vars[i] + (1 - tau) * slow_target_actor_var)
            update_slow_target_ops_a.append(update_slow_target_actor_op)
        self.update_slow_targets_op_a = tf.group(*update_slow_target_ops_a)

    # will use this to initialize both the actor network its slowly-changing target network with same structure
    def generate_actor_network(self, s, schedule, num_agent, obs_dim_agent, trainable):

        return schedule_net.generate_schedulenet(s, schedule, num_agent, obs_dim_agent, trainable)

    def action_for_state(self, state_ph, schedule):

        schedule_vector = schedule_net.schedule_to_vector(schedule)
        return self.sess.run(self.actions, feed_dict={self.state_ph: state_ph,
                                                      self.is_training_ph: False,
                                                      self.schedule_ph: schedule_vector})

    def training_actor(self, state_ph, action_ph, td_errors, schedule):

        schedule_vector = schedule_net.schedule_to_vector(schedule)
        return self.sess.run(self.actor_train_op,
                             feed_dict={self.state_ph: state_ph,
                                        self.action_ph: action_ph,
                                        self.td_errors: td_errors,
                                        self.is_training_ph: True,
                                        self.schedule_ph: schedule_vector})

    # def training_target_actor(self):
    #     return self.sess.run(self.update_slow_targets_op_a,
    #                          feed_dict={self.is_training_ph: False,
    #                                     self.schedule: self.conn_i})


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
        self.action_ph = tf.placeholder(dtype=tf.int32, shape=[None])
        self.a_onehot = tf.one_hot(self.action_ph, action_dim, 1.0, 0.0)

        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])

        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.st_action_ph = tf.placeholder(dtype=tf.int32, shape=[None])
        self.sta_onehot = tf.one_hot(self.st_action_ph, action_dim, 1.0, 0.0)

        self.is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None])  # indicators (go into target computation)
        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())  # for dropout

        with tf.variable_scope(scope):
            # Critic applied to state_ph
            self.q_values = self.generate_critic_network(self.state_ph, self.a_onehot, trainable=True)

        # slow target critic network
        with tf.variable_scope('slow_target_'+scope):
            self.slow_q_values = tf.stop_gradient(
                self.generate_critic_network(self.next_state_ph, self.sta_onehot, trainable=False))

        # One step TD targets y_i for (s,a) from experience replay
        # = r_i + gamma*Q_slow(s',mu_slow(s')) if s' is not terminal
        # = r_i if s' terminal
        targets = tf.expand_dims(self.reward_ph, 1) + tf.expand_dims(self.is_not_terminal_ph, 1) * gamma * self.slow_q_values

        # 1-step temporal difference errors
        self.td_errors = targets - self.q_values

        critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        critic_loss = tf.reduce_mean(tf.square(self.td_errors))

        # critic optimizer
        self.critic_train_op = tf.train.AdamOptimizer(lr_critic * lr_decay).minimize(critic_loss, var_list=critic_vars)

        slow_target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_'+scope)
        update_slow_target_ops_c = []
        for i, slow_target_var in enumerate(slow_target_critic_vars):
            update_slow_target_critic_op = slow_target_var.assign(tau * critic_vars[i] + (1 - tau) * slow_target_var)
            # update_slow_target_critic_op = slow_target_var.assign(critic_vars[i]) #copy only
            update_slow_target_ops_c.append(update_slow_target_critic_op)
        self.update_slow_targets_op_c = tf.group(*update_slow_target_ops_c)

    # will use this to initialize both the critic network its slowly-changing target network with same structure
    def generate_critic_network(self, s, a, trainable):
        if FLAGS.use_action_in_critic:
          state_action = tf.concat([s, a], axis=1)
        else:
          state_action = s

        hidden = tf.layers.dense(state_action, h1_critic, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                 bias_initializer=tf.constant_initializer(0.1),  # biases
                                 use_bias=True, trainable=trainable, name='dense_c1')

        hidden_2 = tf.layers.dense(hidden, h2_critic, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                 bias_initializer=tf.constant_initializer(0.1),  # biases
                                 use_bias=True, trainable=trainable, name='dense_c2')

        hidden_3 = tf.layers.dense(hidden_2, h3_critic, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                 bias_initializer=tf.constant_initializer(0.1),  # biases
                                 use_bias=True, trainable=trainable, name='dense_c3')

        q_values = tf.layers.dense(hidden_3, 1, trainable=trainable,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                                   bias_initializer=tf.constant_initializer(0.1),  # biases
                                   name='dense_c4',use_bias=False)
        return q_values

    def training_critic(self, state_ph, action_ph, reward_ph, next_state_ph, st_action_ph, is_not_terminal_ph):

        return self.sess.run([self.td_errors, self.critic_train_op],
                             feed_dict={self.state_ph: state_ph,
                                        self.action_ph: action_ph,
                                        self.reward_ph: reward_ph,
                                        self.next_state_ph: next_state_ph,
                                        self.st_action_ph: st_action_ph,
                                        self.is_not_terminal_ph: is_not_terminal_ph,
                                        self.is_training_ph: True})

    def training_target_critic(self):
        return self.sess.run(self.update_slow_targets_op_c,
                             feed_dict={self.is_training_ph: False})

    def get_critic_q(self, state_ph, action_ph):

        return self.sess.run([self.q_values],
                             feed_dict={self.state_ph: state_ph,
                                        self.action_ph: action_ph,
                                        self.is_training_ph: True})


class SchedulerNetwork:
    def __init__(self, sess, obs_dim, n_player):

        self.sess = sess
        self.obs_dim = obs_dim
        self.n_player = n_player

        # placeholders
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim * self.n_player])
        self.schedule_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.n_player])

        self.td_errors = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=())  # for dropout

        # actor network
        with tf.variable_scope('schedule'):
            self.schedule_policy = self.generate_scheduler(self.obs_ph, trainable=True)

        # actor loss function (mean Q-values under current policy with regularization)
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='schedule')
        self.responsible = tf.multiply(self.schedule_policy[0], self.schedule_ph)  # =\pi (policy)
        log_prob = tf.log(1e-10+tf.reduce_sum(self.responsible, reduction_indices=1, keep_dims=True))
        entropy = -tf.reduce_sum(self.schedule_policy[0] * tf.log(1e-10+self.schedule_policy[0]), 1)
        self.loss = tf.reduce_sum(-(tf.multiply(log_prob, self.td_errors) + 0.01 * entropy))

        var_grads = tf.gradients(self.loss, self.actor_vars)
        self.grad = var_grads
        self.scheduler_train_op = tf.train.AdamOptimizer(lr_actor * lr_decay).apply_gradients(
            zip(var_grads, self.actor_vars))

    def generate_scheduler(self, obs, trainable=True):
        obs_list = list()
        sched_list = list()

        for i in range(self.n_player):
            obs_list.append(obs[:, i * self.obs_dim:(i + 1) * self.obs_dim])

        for i in range(self.n_player):
            s = self.generate_schedule_network(obs_list[i], trainable)
            sched_list.append(s)

        schedule = tf.concat(sched_list, axis=-1)

        schedule_softmax = tf.nn.softmax(schedule)

        return schedule_softmax, schedule

    def generate_schedule_network(self, obs, trainable=True):
        hidden_1 = tf.layers.dense(obs, h1_scheduler, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                                   bias_initializer=tf.constant_initializer(0.1),  # biases
                                   use_bias=True, trainable=trainable)
        hidden_2 = tf.layers.dense(hidden_1, h2_scheduler, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                                   bias_initializer=tf.constant_initializer(0.1),  # biases
                                   use_bias=True, trainable=trainable)

        hidden_3 = tf.layers.dense(hidden_2, h3_scheduler, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                                   bias_initializer=tf.constant_initializer(0.1),  # biases
                                   use_bias=True, trainable=True)

        schedule = tf.layers.dense(hidden_3, 1, trainable=trainable)

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
