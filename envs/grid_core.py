import numpy as np
import config


COLOR_TO_IDX = config.COLOR_TO_IDX
OBJECT_TO_IDX = config.OBJECT_TO_IDX
IDX_TO_OBJECT = config.IDX_TO_OBJECT

N = 0
E = 1
O = 2
W = 3
S = 4

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self, itype, color):
        assert itype in OBJECT_TO_IDX, itype
        assert color in COLOR_TO_IDX, color
        self.itype = itype
        self.color = color
        self.contains = None

        # name 
        self.name = ''
        # properties:
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0

    @property
    def pos(self):
        return self._x, self._y

    def set_pos(self, x, y):
        self._x = x
        self._y = y

# properties of agent entities
class CoreAgent(Entity):
    def __init__(self, itype='agent', color='green'):
        super(CoreAgent, self).__init__(itype, color)
        self.name = ""
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = True
        # action
        self.action = Action()
        # if waiting for other agents action
        self.waiting = False
        # if done doing its action in the current step
        self.done_moving = False
        # if the intended step collided 
        self.collided = False

        self._obs = None
        self._x = 0
        self._y = 0
        self.obs_range = 1
        # the agent can partially observe the grid by default
        self.fully_observable = False

    def update_obs(self, obs):
        self._obs = obs

    def get_obs(self):
        return self._obs

class Wall(Entity):
    def __init__(self, color='grey'):
        super(Wall, self).__init__('wall', color)

class Grid(object):
    """
    Represent a grid and operations on it
    """

    def __init__(self, width, height):
        assert width >= 2
        assert height >= 2

        self.width = width
        self.height = height
        self.reset()

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        if ((i >= 0 and i < self.width) and \
            (j >= 0 and j < self.height)):
            return self.grid[j * self.width + i]

        return Wall()

    def reset(self):
        self.grid = [None] * self.width * self.height

    def setHorzWall(self, x, y, length=None):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, Wall())

    def setVertWall(self, x, y, length=None):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, Wall())

    def wallRect(self, x, y, w, h):
        self.setHorzWall(x, y, w)
        self.setHorzWall(x, y+h-1, w)
        self.setVertWall(x, y, h)
        self.setVertWall(x+w-1, y, h)

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    def encode(self):
        """
        Produce a compact numpy encoding of the grid
        """

        array = np.zeros(shape=(self.height, self.width, 3), dtype=np.int8)

        for j in range(0, self.height):
            for i in range(0, self.width):

                v = self.get(i, j)
                if isinstance(v, CoreAgent):
                    array[j, i, 2] = v.id

                if v == None:
                    continue


                array[j, i, 0] = OBJECT_TO_IDX[v.itype]
                array[j, i, 1] = COLOR_TO_IDX[v.color]

        return array


# multi-agent world
class World(object):
    def __init__(self, width, height):
        # list of agents and entities (can change at execution-time!)
        self.agents = []

        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2

        self.width = width
        self.height = height

        self.grid = Grid(self.width, self.height)
        self.grid.wallRect(0, 0, self.width, self.height)

        self.step_cnt = 0

    def empty_grid(self):
        self.step_cnt = 0
        self.grid.reset()

    def placeObj(self, obj, top=None, size=None, reject_fn=None):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to randomly place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)

        if size is None:
            size = (self.grid.width, self.grid.height)

        while True:
            pos = (
                np.random.randint(top[0], top[0] + size[0]),
                np.random.randint(top[1], top[1] + size[1])
            )

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)
        obj.set_pos(pos[0], pos[1])
        return pos

    def single_agent_step(self, agent, action):
        if agent.done_moving or agent.waiting:
            return

        x, y = agent.pos
        action = agent.action.u

        if   action == N:
            y -= 1
        elif action == E:
            x -= 1
        elif action == W:
            x += 1
        elif action == S:
            y += 1
        elif action == O:
            agent.done_moving = True
            agent.collided = False
            return

        intended_cell = self.grid.get(x, y)
        if isinstance(intended_cell, CoreAgent):
            agent.waiting = True
            # let the other agent move first
            self.single_agent_step(intended_cell, intended_cell.action.u)
            agent.waiting = False
            # get the intended cell (to check if it is empty)
            intended_cell = self.grid.get(x, y)

        # check if the intended cell is empty
        if not intended_cell is None:  
            agent.collided = True
        else:
            x_0, y_0 = agent.pos
            self.grid.set(x_0, y_0, None)
            self.grid.set(x, y, agent)
            agent.set_pos(x, y)
            agent.collided = False

        agent.done_moving = True

    # update state of the world
    def step(self, action_n):
        self.step_cnt += 1
        # set the action
        for i, agent in enumerate(self.agents):
            agent.action.u = action_n[i]
            agent.done_moving = False
            
        # do the action
        for agent in self.agents:
            self.single_agent_step(agent, agent.action.u)

        # update observations of all agents
        self.set_observations()

    def set_observations(self):
        for agent in self.agents:
            if agent.fully_observable:
                obs = self.grid
            else:
                x, y = agent.pos
                r = agent.obs_range
                obs = self.grid.slice(x-r, y-r,r*2+1,r*2+1)
            agent.update_obs(self.encode_grid_as_state(obs))

    def get_global_state(self):
        return self.encode_grid_as_state(self.grid)

    def encode_grid_as_state(self, grid):
        encoded = grid.encode() # full encoded map
        # state representation plan: one-hot vector per grid cell
        # id-th index marked when any kind of agent is there
        # 0-th index marked when there is a wall
        n = len(self.agents) # number of agents

        res = np.array([])
        for cell in encoded.reshape(-1, 3):
            cell_onehot = np.zeros(n + 1)
            if IDX_TO_OBJECT[cell[0]] == 'wall':
                cell_onehot[0] = 1.0
            elif cell[0] != 0:
                cell_onehot[cell[2]] = 1.0
            res = np.concatenate((res, cell_onehot))

        return res