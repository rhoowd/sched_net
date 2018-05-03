from __future__ import print_function
from __future__ import division
from agents.replay_buffer import ReplayBuffer
from agents.ind_ac.agent import Agent
from agents.simple_agent import StaticAgent as NonLearningAgent
import tensorflow as tf
import numpy as np
import config
import sys

FLAGS = config.flags.FLAGS

training_step = FLAGS.training_step
testing_step = FLAGS.testing_step
pre_train_steps = FLAGS.pre_train_steps
max_step = FLAGS.max_step
save_file = "results/nn/nn-" + config.file_name
epsilon_dec = 1.0/training_step
epsilon_min = 0.1

load_flag = FLAGS.load_nn                       # Use a pre-trained network
load_file = FLAGS.nn_file                       # Filename of the weights to be loaded

class Trainer():
    def __init__(self, env):
        self._env = env
        self._n_predator = FLAGS.n_predator
        self._n_prey = FLAGS.n_prey
        self._agent_profile = self._env.get_agent_profile()
        self._agent_precedence = self._env.agent_precedence
        self.sess = tf.Session()

        self._agents = []
        for i, atype in enumerate(self._agent_precedence):
            if atype == "predator":
                agent = Agent(self._agent_profile["predator"]["act_dim"],
                              self._agent_profile["predator"]["obs_dim"][0],
                              self.sess, name="predator_" + str(i))
            else:
                agent = NonLearningAgent(2)
                # agent = NonLearningAgent(self._agent_profile[atype]["act_dim"])

            self._agents.append(agent)

        # Create replay buffer (not included in the agents coz we might use central RB)
        self._replay_buffer = []
        for atype in self._agent_precedence:
            if atype == "predator":
                self._replay_buffer.append(ReplayBuffer())
            else:
                self._replay_buffer.append(None)

        # intialize tf variables
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if load_flag:
            self.saver.restore(self.sess, "results/nn/" + load_file)

        self.epsilon = 1.0

    def get_actions(self, obs_n, step, train=True):  
        """
        Get the actions from each agent of a centrain type

        :param obs_n: the observations
        :param step: the current step
        :param train: if training
        :return: actions of each [atype] agent
        """
        act_n = []
        for i, agent in enumerate(self._agents):
            if isinstance(agent, NonLearningAgent):
                act_n.append(agent.act(None)) # let it do its own random action
                continue

            if (train and (step < pre_train_steps or np.random.rand() < self.epsilon)):

                # guided action (no collision)
                shape = int(np.sqrt(obs_n[i].shape[0]/(FLAGS.history_len)))
                imap = obs_n[i].reshape((FLAGS.history_len,shape,shape))
                minimap = imap[-1,:,:]

                valid_act = [2]
                center = shape//2
                if minimap[center-1,center] == 0: # up
                    valid_act.append(0)
                if minimap[center,center-1] == 0: #left
                    valid_act.append(1)
                if minimap[center,center+1] == 0: #right
                    valid_act.append(3)
                if minimap[center+1,center] == 0: #down
                    valid_act.append(4)
                action = np.random.choice(valid_act)
                act_n.append(action)
                continue
            
            qval = agent.act(obs_n[i])
            
            if np.isnan(qval).any():
                print("Value Error: nan")
                print(qval)
                sys.exit()

            if train:
                act_n.append(np.random.choice(len(qval[0]),p=qval[0]))
            else:
                act_n.append(np.random.choice(len(qval[0]),p=qval[0]))
                # act_n.append(np.argmax(qval))
                print(qval)

        return np.array(act_n, dtype=np.int32)

    def train_agents_network(self, minibatch, step):
        """
        Update parameters of each agent of a certain type

        :param minibatch: the data to be trained on
        :param step: the current step
        """
        for i, agent in enumerate(self._agents):
            agent.train(minibatch[i], step)

    def learn(self):
        step = 0
        episode = 0

        while step < training_step:
            episode += 1
            obs_n = self._env.reset()
            total_reward = np.zeros(FLAGS.n_predator)

            print("===== episode %d =====" %(episode))

            # for ep_step in xrange(1, max_step+1):
            ep_step = 0
            while True:
                step += 1 # increment global step
                ep_step += 1
                act_n = self.get_actions(obs_n, step)

                obs_n_next, reward_n, done_n, info_n  = self._env.step(act_n)

                # if (reward_n > 0).all():
                #     reward_n = [200 - ep_step, 200 - ep_step]
                #     done = True
                # elif ep_step == max_step:
                #     done = True
                    # reward_n = [-1, -1]

                # if ep_step == max_step and not done:
                    # reward_n = [-1]
                total_reward += np.array(reward_n)[self._agent_profile["predator"]["idx"]]

                for i, rb in enumerate(self._replay_buffer):
                    if not rb is None:
                        exp = (obs_n[i], act_n[i], reward_n[i], obs_n_next[i], 1.0*(not done_n[i]))
                        rb.add_to_memory(exp)
                    
                if step > pre_train_steps:
                    minibatch = []
                    for i in range(len(self._agents)):
                        if self._replay_buffer[i] is None:
                            minibatch.append(None)
                        else:
                            minibatch.append(self._replay_buffer[i].sample_from_memory())

                    self.train_agents_network(minibatch, step)

                self.epsilon = max(self.epsilon - epsilon_dec, epsilon_min)
                obs_n = obs_n_next

                if step % 10000 == 0:
                    self.saver.save(self.sess, save_file, step)

                if step == training_step or sum(done_n)>0:
                    break
            
            print(step, ep_step, reward_n, total_reward, self.epsilon)

        self.test()

    def test(self):
        render = True
        capture_count = 0
        ep_length = []
        for episode in range(5):
            obs_n = self._env.reset()

            print("======================")
            print("===== episode %d =====" %(episode))
            print("======================")

            total_reward = np.zeros(FLAGS.n_predator)
            for ep_step in xrange(1, testing_step+1):
                full_map = self._env.get_full_encoding()[:,:,0]
                act_n = self.get_actions(obs_n, 0, train=False)
                obs_n_next, reward_n, done_n, info_n = self._env.step(act_n)

                pred_obs = [obs_n[i] for i in self._agent_profile["predator"]["idx"]]
                shape = int(np.sqrt(pred_obs[0].shape[0]/(FLAGS.history_len)))
                minimap = np.array(pred_obs).reshape((self._n_predator, FLAGS.history_len,shape,shape))

                print(full_map)
                print(minimap[0, -1], act_n[0], reward_n[0])
                print(minimap[1, -1], act_n[1], reward_n[1])

                obs_n = obs_n_next
                total_reward += np.array(reward_n)[self._agent_profile["predator"]["idx"]]

                if ep_step % 15 == 0:
                    print(ep_step, reward_n, total_reward)

                if sum(done_n)>0:
                    capture_count += 1
                    break
                    
            ep_length.append(ep_step)
            pred_obs = [obs_n[i] for i in self._agent_profile["predator"]["idx"]]
            shape = int(np.sqrt(pred_obs[0].shape[0]/(FLAGS.history_len)))
            minimap = np.array(pred_obs).reshape((self._n_predator, FLAGS.history_len,shape,shape))

            print(self._env.get_full_encoding()[:,:,0])
            print(minimap[0, -1], act_n[0], reward_n[0])
            print(minimap[1, -1], act_n[1], reward_n[1])

        print("CAPTURE COUNT: ", capture_count)
        print("EPISODE LENGTHS: ", ep_length)