#!/usr/bin/env python
# coding=utf8

import tensorflow as tf
import logging
import time
import envs.config_env as config_env
import agents.config_agents as config_agent

flags = tf.flags

config_env.config_env(flags)
config_agent.config_agent(flags)

# Make result file with given filename
now = time.localtime()
s_time = "%02d%02d%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
file_name = str(flags.FLAGS.n_predator) + "-"
file_name += config_env.get_filename() + "-" + config_agent.get_filename()
file_name += "-" + s_time
result = logging.getLogger('Result')
result.setLevel(logging.INFO)
result_fh = logging.FileHandler("./results/eval/r-" + file_name + ".txt")
result_fm = logging.Formatter('[%(filename)s:%(lineno)s] %(asctime)s\t%(message)s')
result_fh.setFormatter(result_fm)
result.addHandler(result_fh)

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
    'agent'         : 2,
    'predator'      : 3,
    'prey'          : 4
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))