#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from envs.environment import MultiAgentEnv
import envs.scenarios as scenarios
import numpy as np
import config

FLAGS = config.flags.FLAGS


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='pursuit.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback=scenario.done)
    act_n = [2, 2]
    print "action space:", env.action_space[0].n
    print "observation space:", env.observation_space

    obs_n = env.reset()[:2]
    print env.get_agent_profile()
    print env.get_full_encoding()[:, :, 2]
    imap = np.array(obs_n).reshape((2, FLAGS.history_len,3,3,1))

    minimap = imap[:,:,:,:,0]
    print minimap[0, -1]
    print minimap[1, -1]

    while True:
        a0 = input("action of agent 0:")
        a1 = input("action of agent 1:")
        act_n = [a0, a1, 2]
        obs_n, reward_n, done_n, info_n = env.step(act_n)
        obs_n = obs_n[:2]
        

        print env.get_full_encoding()[:,:,2]
        imap = np.array(obs_n).reshape((2, FLAGS.history_len,3,3,1))

        minimap = imap[:,:,:,:,0]
        print minimap[0, -1]
        print minimap[1, -1]


        print reward_n, done_n

