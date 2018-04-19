import numpy as np
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
            agents.append(Agent(itype="predator", color="blue"))
            self.atype_to_idx["predator"].append(i)

        # add preys
        for i in xrange(n_prey):
            agents.append(Prey())
            self.atype_to_idx["prey"].append(n_predator + i)

        world.agents = agents
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
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
        self.prey_captured = False

    def reward(self, agent, world):
        if agent.itype == "predator":
            if self.prey_captured:
                return 1
            else:
                for i in self.atype_to_idx["prey"]:
                    prey = world.agents[i]
                    if prey.cannot_move():
                        print "captured"
                        self.prey_captured = True
                        return 1
        else: # if prey
            if agent.cannot_move():
                return -1

        return 0

    def observation(self, agent, world):
        obs = agent.get_obs().flatten()
        return obs

    def done(self, agent, world):
        return self.prey_captured