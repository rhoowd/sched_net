import numpy as np

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'empty'         : 0,
    'wall'          : 1,
    'door'          : 2,
    'locked_door'   : 3,
    'key'           : 4,
    'ball'          : 5,
    'box'           : 6,
    'goal'          : 7,
    'agent'         : 8
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

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
        self.type = itype
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

    def _set_color(self, r):
        c = COLORS[self.color]
        r.setLineColor(c[0], c[1], c[2])
        r.setColor(c[0], c[1], c[2])

# properties of agent entities
class CoreAgent(Entity):
    def __init__(self, itype='agent', color='green'):
        super(CoreAgent, self).__init__(itype, color)
        # # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # action
        self.action = Action()
        # # script behavior to execute
        # self.action_callback = None
        self._obs = None
        self._x = 0
        self._y = 0
        self.obs_range = 1

    def update_obs(self, obs):
        self._obs = obs

    def get_obs(self):
        return self._obs


class Wall(Entity):
    def __init__(self, color='grey'):
        super(Wall, self).__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

class Grid:
    """
    Represent a grid and operations on it
    """

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height
        self.reset()

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

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

        array = np.zeros(shape=(self.height, self.width, 2), dtype='uint8')

        for j in range(0, self.height):
            for i in range(0, self.width):

                v = self.get(i, j)

                if v == None:
                    continue

                array[j, i, 0] = OBJECT_TO_IDX[v.type]
                array[j, i, 1] = COLOR_TO_IDX[v.color]

        return array

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2

        self.width = 4
        self.height = 5

        self.grid = Grid(self.width, self.height)
        self.grid.wallRect(0, 0, self.width, self.height)

    def get_world_dims(self):
        """
        return width and height of the entire environment
        """
        return self.width, self.height


    def empty_grid(self):
        self.grid.reset()
        self.grid.wallRect(0, 0, self.width, self.height)

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

            # Don't place the object where the agent is
            # if pos == self.start_pos:
                # continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)
        obj.set_pos(pos[0], pos[1])
        return pos

    

    def single_agent_step(self, agent, action):
        x, y = agent.pos

        if  (action == N and not self.grid.get(x, y-1) is None) or \
            (action == E and not self.grid.get(x-1, y) is None) or \
            (action == W and not self.grid.get(x+1, y) is None) or \
            (action == S and not self.grid.get(x, y+1) is None):
            print self.grid.get(x, y-1), x, y-1
            print self.grid.get(x-1, y), x-1, y
            print self.grid.get(x+1, y), x+1, y
            print self.grid.get(x, y+1), x, y+1
            print 'here', action, x, y
            return
        
        else:
            self.grid.set(x, y, None)
        
        if action == N:
            y -= 1
        elif action == E:
            x -= 1
        elif action == W:
            x += 1
        elif action == S:
            y += 1

        self.grid.set(x, y, agent)
        agent.set_pos(x, y)

    # update state of the world
    def step(self, action_n):

        # do the action
        for i, agent in enumerate(self.agents):
            self.single_agent_step(agent, action_n[i])
            agent.action.u = action_n[i]

        self.set_observations()

    def set_observations(self):
        for agent in self.agents:
            x, y = agent.pos
            obs_range = agent.obs_range
            obs = self.grid.slice(x-obs_range, y-obs_range,obs_range*2+1,obs_range*2+1)
            agent.update_obs(obs.encode())

        print self.grid.encode()[:,:,0]
        print self.agents[0].pos
        print self.agents[1].pos