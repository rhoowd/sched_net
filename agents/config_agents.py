#!/usr/bin/env python
# coding=utf8
# import agents


def config_agent(_flags):
    flags = _flags

    flags.DEFINE_string("agent", "schedule", "Agent")

    flags.DEFINE_integer("training_step", 2000, "Training time step")
    flags.DEFINE_integer("testing_step", 1000, "Testing time step")
    flags.DEFINE_integer("max_step", 200, "Maximum time step per episode")
    flags.DEFINE_integer("eval_step", 2500, "Number of steps before training")

    # RL setting
    flags.DEFINE_float("df", 0.999, "Discount factor")

    # DQN
    flags.DEFINE_float("lr", 0.0001, "Learning rate")
    flags.DEFINE_integer("b_size", 10000, "Size of the replay memory")
    flags.DEFINE_integer("m_size", 64, "Minibatch size")
    flags.DEFINE_integer("pre_train_step", 10, "during [m_size * pre_step] take random action")

    # Actor Critic
    flags.DEFINE_float("c_lr", 0.0001, "Learning rate")
    flags.DEFINE_float("a_lr", 0.00001, "Learning rate")
    flags.DEFINE_float("tau", 0.05, "Learning rate")
    flags.DEFINE_boolean("use_action_in_critic", False, "Use guided samples")

    # Scheduling
    flags.DEFINE_float("s_lr", 0.0001, "Learning rate")
    flags.DEFINE_string("schedule", 'schedule', "Scheduing type: schedule, random, connect, disconnect")

    # Basic setting for simulation
    flags.DEFINE_boolean("load_nn", False, "Load nn from file or not")
    flags.DEFINE_string("nn_file", "results/nn/s", "The name of file for loading")
    flags.DEFINE_boolean("train", True, "Training or testing")
    flags.DEFINE_boolean("qtrace", False, "Use q trace")
    flags.DEFINE_boolean("kt", False, "Keyboard input test")


def get_filename():
    import config
    FLAGS = config.flags.FLAGS

    # return "a-" + FLAGS.agent + "-lr-" + str(FLAGS.lr) + "-ms-" + str(FLAGS.m_size)
    return "a-" + FLAGS.agent + "-sc-" + str(FLAGS.schedule) + "-clr-" + str(FLAGS.c_lr) + "-alr-" + str(
        FLAGS.a_lr) + "-slr-" + str(FLAGS.s_lr) + "-ms-" + str(FLAGS.m_size)
