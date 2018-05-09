from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import sys
import random
from agents.schedule_obs.replay_buffer import ReplayBuffer
from agents.schedule_obs.ac_network import ActorNetwork
from agents.schedule_obs.ac_network import CriticNetwork
from agents.schedule_obs.ac_network import SchedulerNetwork
from agents.evaluation import Evaluation
from agents.schedule_obs import schedule_net

import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')


class SchedulingAgent(object):

    def __init__(self, n_agent, action_dim, state_dim, obs_dim, name=""):
        logger.info("Schedule")

        self._n_agent = n_agent
        self._state_dim = state_dim
        self._action_dim_per_unit = action_dim
        self._obs_dim_per_unit = obs_dim

        # joint action space for critic network
        self._joint_action_dim = self._action_dim_per_unit ** self._n_agent
        # concatenated observation space
        self._obs_dim = self._obs_dim_per_unit * self._n_agent

        self._name = name
        self.update_cnt = 0

        # Make Actor Critic
        tf.reset_default_graph()
        my_graph = tf.Graph()

        print(state_dim, self._obs_dim_per_unit, self._obs_dim)

        with my_graph.as_default():
            self.sess = tf.Session(graph=my_graph, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
            self._actor = ActorNetwork(self.sess, self._n_agent, self._obs_dim, self._action_dim_per_unit, self._name)
            self._critic = CriticNetwork(self.sess, self._state_dim, self._joint_action_dim, self._name)

            self._scheduler = SchedulerNetwork(self.sess, self._n_agent, self._obs_dim)

            self.sess.run(tf.global_variables_initializer())

        self.replay_buffer = ReplayBuffer()

        self._eval = Evaluation()
        self.q_prev = None

    def explore(self):
        return [random.randrange(self._action_dim_per_unit)
                for _ in range(self._n_agent)]

    def compose_joint_action(self, action_list):
        # compose action list into joint action
        r = 0
        for a in action_list:
            r = a + r * self._action_dim_per_unit
        return r

    def act(self, obs_list, schedule_list):

        # TODO just argmax when testing..

        action_prob_list = self._actor.action_for_state(np.concatenate(obs_list).reshape(1, self._obs_dim),
                                                        schedule_list.reshape(1, self._n_agent))
        if np.isnan(action_prob_list).any():
            raise ValueError('action_prob contains NaN')
        action_list = []
        for action_prob in action_prob_list.reshape(self._n_agent, self._action_dim_per_unit):
            action_list.append(np.random.choice(len(action_prob), p=action_prob))

        return action_list

    def schedule(self, obs_list):

        # pick one agent to communicate
        # TODO generalize for the number of senders

        schedule_prob = self._scheduler.schedule_for_obs(np.concatenate(obs_list)
                                                           .reshape(1, self._obs_dim))
        schedule_idx = np.random.choice(self._n_agent, p=schedule_prob[0])
        ret = np.zeros(self._n_agent)
        ret[schedule_idx] = 1.0
        return ret

    def train(self, state, obs_list, action_list, reward_list, state_next, obs_next_list, schedule_list, done):

        s = state
        o = obs_list
        a = action_list
        r = np.sum(reward_list)
        s_ = state_next
        o_ = obs_next_list
        c = schedule_list

        self.store_sample(s, o, a, r, s_, o_, c, done)
        self.update_ac()

        # a = self.action_to_index(action[0], action[1])
        # a_h = self.action_to_nhot(a, 10)
        # s = self.state_to_index(state)
        # s_n = self.state_to_index(state_n)
        # r = np.sum(reward)
        # o = self.obs_to_onehot(obs)
        # c = schedule
        # c_i = self.schedule_to_onehot(schedule)

        # self.store_sample(s, a, r, s_n, done, o, a_h, c, c_i)
        # self.update_ac()

        # return 0

    def store_sample(self, s, o, a, r, s_, o_, c, done):

        self.replay_buffer.add_to_memory((s, o, a, r, s_, o_, c, done))

    def update_ac(self):
        # if FLAGS.qtrace:
        #     self.update_cnt += 1
        #     if self.update_cnt % 2500 == 0:
        #         self.q()

        if len(self.replay_buffer.replay_memory) < 10 * FLAGS.m_size:
            return 0

        minibatch = self.replay_buffer.sample_from_memory()

        s, o, a, r, s_, o_, c, d = map(np.array, zip(*minibatch))

        a_concat = a
        a_joint = np.apply_along_axis(self.compose_joint_action, 1, a)
        o = np.reshape(o, [-1, self._obs_dim])
        o_ = np.reshape(o_, [-1, self._obs_dim])

        if FLAGS.schedule == 'schedule':
            td_error, _ = self._critic.training_critic(s, a_joint, r, s_, a_joint, d)  # train critic
            _ = self._actor.training_actor(o, a_concat, td_error, c)  # train actor
            _ = self._scheduler.training_scheduler(o, c, td_error)
            _ = self._critic.training_target_critic()  # train slow target critic

        else:
            td_error, _ = self._critic.training_critic(s, a_joint, r, s_, a_joint, d)  # train critic
            _ = self._actor.training_actor(o, a_concat, td_error, c)  # train actor
            _ = self._critic.training_target_critic()  # train slow target critic


    # def schedule_to_onehot(self, schedule):

    #     ret = np.zeros(self._n_predator)
    #     for i in range(self._n_predator):
    #         if schedule[i]:
    #             ret[i] = 1.0

    #     return ret

    # def obs_to_onehot(self, obs):
    #     ret = list()

    #     for a in range(self._n_predator):
    #         wall = np.zeros(self._obs_dim)
    #         predator = np.zeros(self._obs_dim)
    #         prey = np.zeros(self._obs_dim)

    #         for i in range(self._obs_dim):
    #             if obs[a][i] == 1:
    #                 wall[i] = 1
    #             elif obs[a][i] == 3:
    #                 predator[i] = 1
    #             elif obs[a][i] == 4:
    #                 prey[i] = 1

    #         ret.append(np.concatenate([wall, predator, prey]))

    #     ret = np.concatenate(ret)
    #     return ret

    # def state_to_index(self, state):
    #     """
    #     For the single agent case, the state is only related to the position of agent 1
    #     :param state:
    #     :return:
    #     """
    #     # p1, p2 = self.get_predator_pos(state)
    #     p1 = self.get_pos_by_id(state, 1)
    #     p2 = self.get_pos_by_id(state, 2)
    #     prey = self.get_pos_by_id(state, 3)

    #     ret = np.zeros(self._state_dim)
    #     ret[p1] = 1.0
    #     ret[p2 + self._state_dim_single] = 1.0
    #     ret[prey + 2*self._state_dim_single] = 1.0

    #     return ret

    # def get_predator_pos(self, state):
    #     """
    #     return position of agent 1 and 2
    #     :param state: input is state
    #     :return:
    #     """
    #     state_list = list(np.array(state).ravel())
    #     return state_list.index(1), state_list.index(2)

    # def get_pos_by_id(self, state, id):
    #     state_list = list(np.array(state).ravel())
    #     return state_list.index(id)

    # def onehot(self, index, size):
    #     n_hot = np.zeros(size)
    #     n_hot[index] = 1.0
    #     return n_hot

    # def index_to_action(self, index):
    #     return index % 5, index / 5

    # def action_to_index(self, a1, a2):
    #     return a1 + a2 * 5

    # def action_to_nhot(self, a_i, size):
    #     a1, a2 = self.index_to_action(a_i)
    #     ret = np.zeros(size)

    #     ret[a1] = 1.0
    #     ret[a2+5] = 1.0

    #     return ret

    # def q(self):
    #     q_a = 0
    #     q_value = []
    #     for p1 in range(self.map_size ** 2):
    #         for p2 in range(self.map_size ** 2):
    #             if p1 == p2:
    #                 continue
    #             for prey in range(self.map_size ** 2):
    #                 if prey == p1 or prey == p2:
    #                     continue

    #                 s = np.zeros(self._state_dim)
    #                 s[p1] = 1.0
    #                 s[p2 + self._state_dim_single] = 1.0
    #                 s[prey + 2*self._state_dim_single] = 1.0

    #                 for a in range(25):
    #                     q = self._critic.get_critic_q(s[None], [a])
    #                     q_value.append(q)
    #                     q_a += q[0][0][0]

    #     print self.update_cnt, q_a/len(q_value)
