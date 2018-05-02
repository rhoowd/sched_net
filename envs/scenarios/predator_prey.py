import numpy as np
from envs.grid_core import World, CoreAgent
from envs.scenario import BaseScenario
import config

FLAGS = config.flags.FLAGS


class Prey(CoreAgent):
    def __init__(self, fully_observable=False):
        super(Prey, self).__init__('prey', 'green')
        self.fully_observable = fully_observable
        self._movement_mask = np.array(
            [[0,1,0],
             [1,0,1],
             [0,1,0]], dtype=np.int8
        )

    def is_captured(self, world):
        x, y = self.pos
        minimap = world.grid.slice(x-1,y-1,3,3).encode()[:,:,0] != 0
        return np.sum(minimap*self._movement_mask) == 4

class Predator(CoreAgent):
    def __init__(self, fully_observable=False):
        super(Predator, self).__init__('predator', 'blue')
        self.fully_observable = fully_observable

class Scenario(BaseScenario):
    def __init__(self):
        self.prey_captured = False

    def make_world(self):
        map_size = FLAGS.map_size
        world = World(width=map_size, height=map_size)

        # list of all agents
        agents = []
        self.atype_to_idx = {
            "predator": [],
            "prey": []
        }
        # make agents

        # add predators
        n_predator = FLAGS.n_predator
        for i in range(n_predator):
            agents.append(Predator(fully_observable=False))
            self.atype_to_idx['predator'].append(i)

        # add preys
        n_prey = FLAGS.n_prey
        for i in range(n_prey):
            agents.append(Prey(fully_observable=False))
            self.atype_to_idx['prey'].append(n_predator + i)

        # used by BaseScenario
        # assign id to agents
        world.agents = agents
        for i, agent in enumerate(world.agents):
            agent.id = i + 1
            agent.silent = True

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.empty_grid()

        # randomly place agents
        for agent in world.agents:
            world.placeObj(agent)
        
        world.set_observations()

        self.prey_captured = False

    def reward(self, agent, world):
        if agent.itype == 'predator':
            if self.prey_captured:
                return 1
            else:
                reward = -0.01
                # determine whether the prey has been captured
                for i in self.atype_to_idx['prey']:
                    prey = world.agents[i]
                    if prey.is_captured(world):
                        self.prey_captured = True
                        return 1
                return reward
        else: # if prey
            if agent.is_captured(world):
                return -1
        return 0

    def observation(self, agent, world):
        obs = np.array(agent.get_obs()).flatten()
        return obs

    def done(self, agent, world):
        return self.prey_captured