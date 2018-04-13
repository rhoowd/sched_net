import numpy as np
from multiagent.grid_core import World
from multiagent.grid_core import CoreAgent as Agent
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True 

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.empty_grid()
        w, h = world.get_world_dims()
        
        # randomly place agent
        for agent in world.agents:
            world.placeObj(agent)

        world.set_observations()

    def reward(self, agent, world):
        return 1


    def observation(self, agent, world):
        obs = agent.get_obs()
        return obs