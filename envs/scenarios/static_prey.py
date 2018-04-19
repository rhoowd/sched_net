import numpy as np
from envs.scenarios.pursuit import Scenario as BaseScenario
import config

FLAGS = config.flags.FLAGS
map_size = FLAGS.map_size

class Scenario(BaseScenario):
    def __init__(self):
        super(Scenario, self).__init__()

    def reset_world(self, world):
        world.empty_grid()

        # prey_pos = np.random.choice([map_size - 1, 0], 2)
        prey_pos = [0, 0]
        prey_idx = self.atype_to_idx["prey"][0]
        world.placeObj(world.agents[prey_idx], top=prey_pos, size=(1,1))

        top = ((prey_pos[0]+1)%map_size, (prey_pos[1]+1)%map_size)
        for idx in self.atype_to_idx["predator"]:
            world.placeObj(world.agents[idx], top=top, size=(2,2))

        world.set_observations()

        # fill the history with current observation
        for i in self.atype_to_idx["predator"]:
            world.agents[i].fill_obs()

        self.prey_captured = False
