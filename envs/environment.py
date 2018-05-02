import gym
from gym import spaces
import numpy as np

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!


class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.agents
        # set required vectorized gym env property
        self.n = len(world.agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
      
        # environment parameters
        self.discrete_comm_space = True
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.agent_precedence = []
        for agent in self.agents:
            self.agent_precedence.append(agent.itype)
            total_action_space = []
            u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            total_action_space.append(u_action_space)

            # communication action space
            if self.discrete_comm_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)

            if not agent.silent:
                total_action_space.append(c_action_space)

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = spaces.MultiDiscrete([act_space.n for act_space in total_action_space])
                else: 
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            obs_dim = len(observation_callback(agent, self.world).flatten())
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

    def get_agent_profile(self):
        agent_profile = {}

        for i, agent in enumerate(self.agents):
            if agent.itype in agent_profile:
                agent_profile[agent.itype]['n_agent'] += 1
                agent_profile[agent.itype]['idx'].append(i)
            else:
                if isinstance(self.action_space[i], spaces.Discrete):
                    act_space = self.action_space[i].n
                    com_space = 0
                else:
                    act_space = self.action_space[i].nvec[0]
                    com_space = self.action_space[i].nvec[1]

                agent_profile[agent.itype] = {
                    'n_agent': 1,
                    'idx': [i],
                    'act_dim': act_space,
                    'com_dim': com_space,
                    'obs_dim': self.observation_space[i].shape
                }

        return agent_profile

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}

        self.agents = self.world.agents
        self.world.step(action_n)

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)

        obs_n = []
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    def get_global_state(self):
        return self.world.get_global_state()