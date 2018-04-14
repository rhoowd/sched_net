#!/usr/bin/env python
# coding=utf8
# import agents


def config_agent(_flags):
    flags = _flags

    flags.DEFINE_string("agent", "ind_ac", "Agent")

    flags.DEFINE_integer("training_step", 300000, "Training time step")
    flags.DEFINE_integer("testing_step", 50, "Testing time step")
    flags.DEFINE_integer("max_step", 200, "Maximum time step per episode")
    flags.DEFINE_integer("pre_train_steps", 8, "Number of steps before training")
    
    flags.DEFINE_integer("replay_buffer_capacity", 8, "Size of the replay memory")
    flags.DEFINE_integer("minibatch_size", 8, "Minibatch size")
    flags.DEFINE_integer("rnn_trace_len", 4, "Length of samples for training RNN")
    
    flags.DEFINE_boolean("load_nn", True, "Load nn from file or not")
    flags.DEFINE_string("nn_file", "results/nn/s", "The name of file for loading")
    
    flags.DEFINE_boolean("train", False, "Training or testing")
    flags.DEFINE_boolean("guide", False, "Use guided samples")

def get_filename():
    import config
    FLAGS = config.flags.FLAGS

    return "a-"+FLAGS.agent