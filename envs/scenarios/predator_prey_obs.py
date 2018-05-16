from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import itertools
import numpy as np
from envs.grid_core import World, CoreAgent
from envs.scenario import BaseScenario
import config

FLAGS = config.flags.FLAGS
IDX_TO_OBJECT = config.IDX_TO_OBJECT


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
    def __init__(self, id=0):
        super(Predator, self).__init__('predator', 'blue')
        self.id = id

        if FLAGS.hetero > 0:  # Hetero
            if self.id == 0:
                self.obs_range = 2
            else:
                self.obs_range = 1

        else:  # Homogeneous
            self.obs_range = FLAGS.obs_range

        self.last_obs_x = 0.5
        self.last_obs_y = 0.5

        self.obs_prey_before = False

    def set_obs_prey(self, px, py):
        self.last_obs_x = px
        self.last_obs_y = py

        self.obs_prey_before = True

    def reset_obs_prey(self):
        self.last_obs_x = 0.5
        self.last_obs_y = 0.5
        self.obs_prey_before = False

    def get_obs_prey(self):
        ret = [self.obs_prey_before, self.last_obs_x, self.last_obs_y]

        return ret


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
            agents.append(Predator(i))
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
            if agent.itype == 'predator':
                agent.reset_obs_prey()
        
        world.set_observations()

        self.prey_captured = False

    def reward(self, agent, world):
        if agent.itype == 'predator':
            if self.prey_captured:
                return 1
            else:
                reward = 0.0
                for i in self.atype_to_idx['predator']:
                    pred = world.agents[i]
                    check = self.check_prey(pred, world)[0]
                    if check == 0.0:
                        reward -= 0.1
                if reward == 0.0:
                    self.prey_captured = True
                    return 1
                return reward
        else: # if prey
            if agent.is_captured(world):
                return -1
        return 0

    def encode_grid_to_onehot(self, world, grid):
        encoded = grid.encode() # full encoded map
        # state representation plan: one-hot vector per grid cell
        # id-th index marked when any kind of agent is there
        # 0-th index marked when there is a wall
        n = len(world.agents) # number of agents

        res = np.array([])
        for cell in encoded.reshape(-1, 3):
            cell_onehot = np.zeros(n + 1)
            if IDX_TO_OBJECT[cell[0]] == 'wall':
                cell_onehot[0] = 1.0
            elif cell[0] != 0:
                cell_onehot[cell[2]] = 1.0
            res = np.concatenate((res, cell_onehot))

        return res

    def observation(self, agent, world):

        pos_normal = True
        prey_flag = True
        ret = []
        if pos_normal:
            ret.append(self.get_pos_normal(agent, world))

        if prey_flag:
            c_prey = self.check_prey(agent, world)

            if agent.itype == 'predator':
                if c_prey[0] == 1:
                    agent.set_obs_prey(c_prey[1], c_prey[2])

                ret.append([c_prey[0]])
                ret.append(agent.get_obs_prey())

        ret = np.concatenate(ret)

        return ret

    def get_pos_normal(self, agent, world):
        x, y = agent.pos  # TODO: order has problem
        ret = np.array([x / (world.grid.width-1), y / (world.grid.height-1)])
        return ret

    def check_prey(self, agent, world):

        obs_native = self.encode_grid_to_onehot(world, agent.get_obs())

        check_prey = 0.0
        coor_prey = 0
        cnt = 0
        obs_prey = []
        px = -1.0
        py = -1.0

        for cell in obs_native.reshape(-1, len(world.agents) + 1):
            # one-hot encoded cell w.r.t. agent id
            check = 0.0
            if np.max(cell) != 0:
                idx = np.argmax(cell)
                if idx in [world.agents[i].id for i in self.atype_to_idx['prey']]:
                    check_prey = 1.0
                    check = 1.0
                    coor_prey = cnt
            obs_prey.append(check)
            cnt += 1

        if check_prey == 1.0:
            obs_size = (2*agent.obs_range + 1)
            px = (coor_prey // obs_size) / (obs_size - 1)
            py = (coor_prey % obs_size) / (obs_size - 1)

            if FLAGS.hetero == 2:
                if agent.id in [4]:
                    if not px == 0.5:
                        print(agent.id, px, py)
                        check_prey = 0.0
                        px = -1.0
                        py = -1.0

        return check_prey, px, py

    def info(self, agent, world):
        # info() returns the global state
        coord_as_state = True
        if coord_as_state:
            # encode coordinates into state
            width = world.grid.width
            height = world.grid.height
            encoded = world.grid.encode()[:, :, 2]
            state = np.zeros((len(world.agents), 2)) # n_agents * (x,y)
            for x, y in itertools.product(*map(range, (width, height))):
                if encoded[y, x] != 0:
                    state[encoded[y, x] - 1] = np.array([x/width, y/height])
            return {'state': state.flatten()}
        else:
            return {'state': self.encode_grid_to_onehot(world, world.grid)}

    def done(self, agent, world):
        return self.prey_captured