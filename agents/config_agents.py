#!/usr/bin/env python
# coding=utf8
# import agents


def config_agent(_flags):
    flags = _flags

    flags.DEFINE_string("agent", "dqn_single_onehot", "Agent")

    flags.DEFINE_integer("training_step", 5000, "Training time step")
    flags.DEFINE_integer("testing_step", 1000, "Testing time step")
    flags.DEFINE_integer("max_step", 200, "Maximum time step per episode")
    flags.DEFINE_integer("eval_step", 1000, "Number of steps before training")
    
    flags.DEFINE_integer("b_size", 50000, "Size of the replay memory")
    flags.DEFINE_integer("m_size", 64, "Minibatch size")
    
    flags.DEFINE_boolean("load_nn", False, "Load nn from file or not")
    flags.DEFINE_string("nn_file", "results/nn/s", "The name of file for loading")
    
    flags.DEFINE_boolean("train", True, "Training or testing")
    flags.DEFINE_boolean("guide", False, "Use guided samples")

    flags.DEFINE_float("lr", 0.0001, "Learning rate")
    flags.DEFINE_float("df", 0.999, "Discount factor")

def get_filename():
    import config
    FLAGS = config.flags.FLAGS

    return "a-"+FLAGS.agent+"-lr-"+str(FLAGS.lr)+"-ms-"+str(FLAGS.m_size)