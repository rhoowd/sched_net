import numpy as np
from envs.scenarios.pursuit import Scenario as BaseScenario
import config

FLAGS = config.flags.FLAGS
map_size = FLAGS.map_size

class Scenario(BaseScenario):
    def __init__(self):
        super(Scenario, self).__init__()
        print "Single agent scenario"

    def reset_world(self, world):
        world.empty_grid()

        prey_pos = [0, 0]

        prey_idx = self.atype_to_idx["prey"][0]
        world.placeObj(world.agents[prey_idx], top=prey_pos, size=(1,1))

        top = ((prey_pos[0]+1)%map_size, (prey_pos[1]+1)%map_size)

        world.placeObj(world.agents[0], top=top, size=(2, 2))
        world.placeObj(world.agents[1], top=[0, 1], size=(1, 1))

        world.set_observations()

        # fill the history with current observation
        for i in self.atype_to_idx["predator"]:
            world.agents[i].fill_obs()

        self.prey_captured = False
