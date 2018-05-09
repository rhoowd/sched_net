#!/usr/bin/env python
# coding=utf8


def config_env(_flags):
    flags = _flags

    # Scenario
    flags.DEFINE_string("scenario", "predator_prey", "Scenario")
    flags.DEFINE_integer("n_predator", 2, "Number of predators")
    flags.DEFINE_integer("n_prey", 1, "Number of preys")
    flags.DEFINE_boolean("obs_diagonal", True, "Whether the agent can see in diagonal directions")
    flags.DEFINE_boolean("moving_prey", True, "Whether the prey is moving")
    
    # Observation
    flags.DEFINE_integer("history_len", 1, "How many previous steps we look back")

    # core
    flags.DEFINE_integer("map_size", 3, "Size of the map")
    flags.DEFINE_float("render_every", 1000, "Render the nth episode")

    # GUI
    flags.DEFINE_boolean("gui", False, "Activate GUI")


def get_filename():
    import config
    FLAGS = config.flags.FLAGS

    return "s-"+FLAGS.scenario+"-map-"+str(FLAGS.map_size)+"-mv-"+str(FLAGS.moving_prey)+"-dg-"+str(FLAGS.obs_diagonal)