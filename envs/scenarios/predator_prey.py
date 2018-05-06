import numpy as np
from envs.grid_core import World, CoreAgent
from envs.scenario import BaseScenario
import config

FLAGS = config.flags.FLAGS


class Prey(CoreAgent):
    def __init__(self):
        super(Prey, self).__init__('prey', 'green')
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
    def __init__(self):
        super(Predator, self).__init__('predator', 'blue')
        self.obs_range = 1

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
            agents.append(Predator())
            self.atype_to_idx['predator'].append(i)

        # add preys
        n_prey = FLAGS.n_prey
        for i in range(n_prey):
            agents.append(Prey())
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
        obs_native = np.array(agent.get_obs())
        # encode all predators and preys into same id
        # TODO: try not to distinguish the same kind of agents..
        indistinguish = True
        if indistinguish:
            obs = np.array([])
            for cell in obs_native.reshape(-1, len(world.agents) + 1):
                # one-hot encoded cell w.r.t. agent id
                compact_cell = np.zeros(3) # wall, predator, prey
                if np.max(cell) != 0:
                    idx = np.argmax(cell)
                    if idx == 0: # wall
                        compact_cell[0] = 1.0
                    elif idx in [world.agents[i].id for i in self.atype_to_idx['predator']]:
                        compact_cell[1] = 1.0
                    elif idx in [world.agents[i].id for i in self.atype_to_idx['prey']]:
                        compact_cell[2] = 1.0
                    else:
                        raise Exception('cell has to be wall/predator/prey!')
                obs = np.concatenate([obs, compact_cell])
            return obs
        else:
            return obs_native

    def done(self, agent, world):
        return self.prey_captured