from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import six
import numpy as np
from agents.cc_obs_sa.agent import SAPredatorAgentObs
from agents.simple_agent import RandomAgent
from agents.evaluation import Evaluation
import logging
import config
from envs.gui import canvas

FLAGS = config.flags.FLAGS
logger = logging.getLogger("Agent")
result = logging.getLogger('Result')

training_step = FLAGS.training_step
testing_step = FLAGS.testing_step

epsilon_dec = 1.0/training_step
epsilon_min = 0.1


class Trainer(object):

    def __init__(self, env):
        logger.info("Centralized Actor-Critic Trainer is created")

        self._env = env
        self._eval = Evaluation()
        self._agent_profile = self._env.get_agent_profile()
        self._state_dim = self._env.get_info()[0]['state'].shape[0]
        self._n_predator = self._agent_profile['predator']['n_agent']

        # joint CAC predator agent
        self._predator_agent = SAPredatorAgentObs(n_agent=self._agent_profile['predator']['n_agent'],
                                                  action_dim=self._agent_profile['predator']['act_dim'],
                                                  state_dim=self._state_dim,
                                                  obs_dim=self._agent_profile['predator']['obs_dim'][0])
        # randomly moving prey agent
        self._prey_agent = []
        for _ in range(self._agent_profile['prey']['n_agent']):
            self._prey_agent.append(RandomAgent(5))

        self.epsilon = 0.5

        # For gui
        if FLAGS.gui:
            self.canvas = canvas.Canvas(self._n_predator, 1, FLAGS.map_size)
            self.canvas.setup()

    def learn(self):

        global_step = 0
        episode_num = 0
        print_flag = False

        while global_step < training_step:
            episode_num += 1
            step_in_ep = 0
            obs_n = self._env.reset()  # obs_n
            state = self._env.get_info()[0]['state']
            total_reward = 0

            done = False
            while not done:
                global_step += 1
                step_in_ep += 1

                action_n = self.get_action(obs_n, global_step)

                obs_n_next, reward_n, done_n, info_n = self._env.step(action_n) # obs_n_next
                state_next = info_n[0]['state']

                # self.print_obs(obs_n)

                # if FLAGS.gui:
                #     self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Hello")

                done_single = sum(done_n) > 0
                self.train_agents(state, obs_n, action_n, reward_n, state_next, obs_n_next, done_single)

                obs_n = obs_n_next
                state = state_next
                total_reward += np.sum(reward_n)

                if is_episode_done(done_n, global_step):
                    # if FLAGS.gui:
                    #     self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Hello", True)
                    if print_flag:
                        print("[train_ep %d]" % (episode_num),"\tstep:", global_step, "\tstep_per_ep:", step_in_ep, "\treward", total_reward)
                    done = True

                if global_step % FLAGS.eval_step == 0:
                    self.test(global_step)
                    break

        self._predator_agent.save_nn(global_step)
        self._eval.summarize()

    def print_obs(self, obs):
        for i in range(FLAGS.n_predator):
            print(obs[i])
        print("")

    def check_obs(self, obs):

        check_list = []
        for i in range(FLAGS.n_predator):
            check_list.append(obs[i][2])

        return np.array(check_list)

    def get_action(self, obs_n, global_step, train=True):
        act_n = [0] * len(obs_n)
        self.epsilon = max(self.epsilon - epsilon_dec, epsilon_min)

        # Action of predator
        if train and (global_step < FLAGS.m_size * FLAGS.pre_train_step or np.random.rand() < self.epsilon):  # with prob. epsilon
            # exploration of centralized predator agent
            predator_action = self._predator_agent.explore()
        else:
            predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
            predator_action = self._predator_agent.act(predator_obs)

        for i, idx in enumerate(self._agent_profile['predator']['idx']):
            act_n[idx] = predator_action[i]

        # Action of prey
        for i, idx in enumerate(self._agent_profile['prey']['idx']):
            act_n[idx] = self._prey_agent[i].act(None)

        return np.array(act_n, dtype=np.int32)

    def train_agents(self, state, obs_n, action_n, reward_n, state_next, obs_n_next, done):
        # train predator only
        predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
        predator_action = [action_n[i] for i in self._agent_profile['predator']['idx']]
        predator_reward = [reward_n[i] for i in self._agent_profile['predator']['idx']]
        _ = [obs_n_next[i] for i in self._agent_profile['predator']['idx']]
        self._predator_agent.train(state, predator_obs, predator_action, predator_reward,
                                   state_next, done)

    def test(self, curr_ep=None):

        global_step = 0
        episode_num = 0

        test_flag = FLAGS.kt
        total_reward = 0
        obs_cnt = np.zeros(self._n_predator)

        while global_step < testing_step:
            episode_num += 1
            step_in_ep = 0
            obs_n = self._env.reset()
            state = self._env.get_info()[0]['state']

            if test_flag:
                print("\nInit\n", obs_n[0])

            while True:

                global_step += 1
                step_in_ep += 1

                action_n = self.get_action(obs_n, global_step, False)
                obs_n_next, reward_n, done_n, info_n = self._env.step(action_n)
                state_next = info_n[0]['state']

                obs_cnt += self.check_obs(obs_n_next)

                if FLAGS.gui:
                    self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Hello")

                if test_flag:
                    aa = six.moves.input('>')
                    if aa == 'c':
                        test_flag = False
                    print(action_n)
                    print(obs_n[0])

                obs_n = obs_n_next
                state = state_next
                total_reward += np.sum(reward_n)

                if is_episode_done(done_n, global_step, "test") or step_in_ep > FLAGS.max_step:
                    if FLAGS.gui:
                        self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Hello", True)
                    break

        print("Test result: Average steps to capture: ", curr_ep, float(global_step) / episode_num,
              "\t", float(total_reward) / episode_num, obs_cnt / episode_num)
        self._eval.update_value("test_result", float(global_step)/episode_num, curr_ep)


def is_episode_done(done, step, e_type="train"):

    if e_type == "test":
        if sum(done) > 0 or step >= FLAGS.testing_step:
            return True
        else:
            return False

    else:
        if sum(done) > 0 or step >= FLAGS.training_step:
            return True
        else:
            return False


