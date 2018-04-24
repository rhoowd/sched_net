import numpy as np
from collections import deque
from envs.grid_core import World
from envs.grid_core import CoreAgent as Agent
from envs.scenario import BaseScenario
import config

FLAGS = config.flags.FLAGS

n_predator = FLAGS.n_predator
n_prey = FLAGS.n_prey
map_size = FLAGS.map_size

class Prey(Agent):
    def __init__(self):
        super(Prey, self).__init__("prey", "green")
        self._movement_mask = np.array(
            [[0,1,0],
             [1,0,1],
             [0,1,0]], dtype=np.int8)

    def cannot_move(self):
        minimap = (self._obs[:,:,0] != 0)
        return np.sum(minimap*self._movement_mask)==4

class Predator(Agent):
    def __init__(self):
        super(Predator, self).__init__("predator", "blue")
        self._obs = deque(maxlen=FLAGS.history_len)
        self.obs_range = 1

    def can_observe_prey(self):
        shape = np.shape(self._obs)
        obs_size = shape[1]*shape[2]
        obs = np.reshape(self._obs, obs_size)
        ret = np.shape(np.where(obs == 4))[1] > 0
        return ret

    def update_obs(self, obs):
        self._obs.append(obs[:,:,0]) # use only the first channel

    def fill_obs(self):
        # fill the whole history with the current observation
        for i in range(FLAGS.history_len-1):
            self._obs.append(self._obs[-1])

class Scenario(BaseScenario):
    def __init__(self):
        self.prey_captured = False

    def make_world(self):
        world = World(width=map_size, height=map_size)

        agents = []
        self.atype_to_idx = {
            "predator": [],
            "prey": []
        }

        # add predators
        for i in xrange(n_predator):
            agents.append(Predator())
            self.atype_to_idx["predator"].append(i)

        # add preys
        for i in xrange(n_prey):
            agents.append(Prey())
            self.atype_to_idx["prey"].append(n_predator + i)

        world.agents = agents
        for i, agent in enumerate(world.agents):
            agent.id = i + 1
            agent.silent = True 

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.empty_grid()

        # randomly place agent
        for agent in world.agents:
            world.placeObj(agent)

        world.set_observations()

        # fill the history with current observation
        for i in self.atype_to_idx["predator"]:
            world.agents[i].fill_obs()

        self.prey_captured = False

    def reward(self, agent, world):
        if agent.itype == "predator":
            if self.prey_captured:
                # return max(10 - world.step_cnt, 0)
                return 1
            else:
                reward = -0.01
                for i in self.atype_to_idx["prey"]:
                    prey = world.agents[i]
                    if prey.cannot_move():
                        # print "captured"
                        self.prey_captured = True
                        reward = 1
                        return reward
                # kdw - Use this for large map size
                # if agent.can_observe_prey():
                #     reward = 0.0
                return reward
        else: # if prey
            if agent.cannot_move():
                return -1

        return 0

    def observation(self, agent, world):
        # print agent.get_obs.shape
        obs = np.array(agent.get_obs()).flatten()
        return obs

    def done(self, agent, world):
        return self.prey_captured