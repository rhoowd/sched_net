from agents.replay_buffer import ReplayBuffer
from agents.ind_ac.agent import Agent
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
        self.sess = tf.Session()

        # Create Agents (use dictionary because I will extend this to heterogenous agents)
        self._agents = {}
        self._agents["predator"] = []
        for i in range(self._n_predator):
            agent = Agent(env, self._agent_profile["predator"]["act_dim"],
                          self._agent_profile["predator"]["obs_dim"],
                          self.sess, name="predator_" + str(i))
            self._agents["predator"].append(agent)

        # Create replay buffer (not included in the agents coz we might use central RB)
        self._replay_buffer = {}
        self._replay_buffer["predator"] = []
        for i in range(self._n_predator):
            self._replay_buffer["predator"].append(ReplayBuffer())

        # intialize tf variables
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if load_flag:
            self.saver.restore(self.sess, "results/nn/" + load_file)

        self.epsilon = 1.0

    def get_actions(self, obs_n, atype, step, train=True):  
        """
        Get the actions from each agent of a centrain type

        :param obs_n: the observations
        :param atype: the type of agent
        :param step: the current step
        :param train: if training
        :return: actions of each [atype] agent
        """
        act_n = []
        for i in range(self._agent_profile[atype]["n_agent"]):
            if (train and (step < pre_train_steps or np.random.rand() < self.epsilon)):
                shape = int(np.sqrt(obs_n[i].shape[0]/(FLAGS.history_len*3)))
                imap = obs_n[i].reshape((FLAGS.history_len,shape,shape,3))
                minimap = imap[:,:,:,0] - imap[:,:,:,2]
                minimap = minimap[-1]

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
                if minimap[center-1,center] == -1: #a_up
                    valid_act.append(5)
                if minimap[center,center-1] == -1: #a_left
                    valid_act.append(6)
                if minimap[center,center+1] == -1: #a_right
                    valid_act.append(7)
                if minimap[center+1,center] == -1: #a_down
                    valid_act.append(8)
                action = np.random.choice(valid_act)
                # action = np.random.randint(self._agent_profile[atype]["act_dim"])
                act_n.append(action)
            else:
                qval = self._agents[atype][i].act(obs_n[i])
                
                if np.isnan(qval).any():
                    print "Value Error: nan"
                    print qval
                    sys.exit()

                act_n.append(np.argmax(qval))

                if not train:
                    print qval


        return np.array(act_n, dtype=np.int32)

    def train_agents_network(self, minibatch, atype, step):
        """
        Update parameters of each agent of a certain type

        :param minibatch: the data to be trained on
        :param atype: the type of agent to train
        :param step: the current step
        """
        for i in range(self._agent_profile[atype]["n_agent"]):
            self._agents[atype][i].train(minibatch[i], step)

    def learn(self):
        step = 0
        episode = 0

        while step < training_step:
            episode += 1
            self._env.reset()
            obs_n = self._env.get_obs()
            obs_n = obs_n["predator"]

            render = (episode % FLAGS.render_every == 0)
            total_reward = np.zeros(FLAGS.n_predator)

            print "===== episode %d =====" %(episode)

            for ep_step in xrange(1, max_step+1):
                step += 1 # increment global step

                act_n = {"predator": self.get_actions(obs_n, "predator", step),
                         "prey": np.array((4), dtype=np.int32)}
                         # "prey": np.random.randint(self._agent_profile["prey"]["act_dim"], size=(1), dtype=np.int32)}

                obs_n_next, reward_n, done = self._env.step(act_n, render)


                act_n = act_n["predator"]
                reward_n = reward_n["predator"]
                obs_n_next = obs_n_next["predator"]

                if (reward_n > 0).all():
                    reward_n = [200 - ep_step, 200 - ep_step]
                    done = True
                elif ep_step == max_step:
                    done = True
                    # reward_n = [-1, -1]

                # if ep_step == max_step and not done:
                    # reward_n = [-1]

                total_reward += reward_n

                # for atype in self._learning_agents():
                for i in range(self._n_predator):
                    exp = (obs_n[i], act_n[i], reward_n[i], obs_n_next[i], 1.0*(not done))
                    self._replay_buffer["predator"][i].add_to_memory(exp)
                    
                if step > pre_train_steps:
                    minibatch = []
                    for i in range(self._n_predator):
                        minibatch.append(self._replay_buffer["predator"][i].sample_from_memory())

                    self.train_agents_network(minibatch, "predator", step)

                self.epsilon = max(self.epsilon - epsilon_dec, epsilon_min)
                obs_n = obs_n_next

                if step % 10000 == 0:
                    self.saver.save(self.sess, save_file, step)

                if step == training_step or done:
                    break
            
            print step, ep_step, reward_n, total_reward, self.epsilon

        self.test()

    def test(self):
        render = True
        for episode in range(5):
            self._env.reset()
            obs_n = self._env.get_obs()
            obs_n = obs_n["predator"]

            total_reward = [0,0] # lipat mo sa envs

            print "======================"
            print "===== episode %d =====" %(episode)
            print "======================"

            total_reward = np.zeros(FLAGS.n_predator)
            for ep_step in xrange(1, testing_step+1):
                act_n = {"predator": self.get_actions(obs_n, "predator", 0, False),
                         "prey": np.array((4), dtype=np.int32)}
                         # "prey": np.random.randint(self._agent_profile["prey"]["act_dim"], size=(1), dtype=np.int32)}
                obs_n_next, reward_n, done = self._env.step(act_n, render)
                shape = int(np.sqrt(obs_n[0].shape[0]/(FLAGS.history_len*3)))
                imap = obs_n.reshape((self._n_predator, FLAGS.history_len,shape,shape,3))

                minimap = imap[:,:,:,:,0] - imap[:,:,:,:,2] - imap[:,:,:,:,1]*2
                print minimap[0, -1], act_n["predator"][0], reward_n["predator"][0]
                print minimap[1, -1], act_n["predator"][1], reward_n["predator"][1]

                obs_n = obs_n_next["predator"]
                reward_n = reward_n["predator"]
                total_reward += reward_n

                if ep_step % 15 == 0:
                    print ep_step, reward_n, total_reward

                if done:
                    break
