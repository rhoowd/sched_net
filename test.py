#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import MultiAgentEnv
# from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

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
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None)
    act_n = [2, 2]
    print "action space:", env.action_space[0].n
    print "observation space:", env.observation_space[0].shape

    obs_n = env.reset()
    print obs_n[0][:,:,0]
    print obs_n[1][:,:,0]

    while True:
        a0 = input("action of agent 0:")
        a1 = input("action of agent 1:")
        act_n = [a0, a1]
        obs_n, reward_n, done_n, info_n = env.step(act_n)
        print obs_n[0][:,:,0]
        print obs_n[1][:,:,0]

        print reward_n, done_n
