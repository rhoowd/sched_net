#!/usr/bin/env python
# coding=utf8

def config_env(_flags):
    flags = _flags

    # Scenario
    flags.DEFINE_string("scenario", "pursuit", "Scenario")
    flags.DEFINE_integer("n_predator", 2, "Number of predators")
    flags.DEFINE_integer("n_prey", 1, "Number of preys")

    # Observation
    # flags.DEFINE_integer("history_len", 1, "How many previous steps we look back")
    flags.DEFINE_integer("history_len", 1, "How many previous steps we look back")

    # core
    flags.DEFINE_integer("map_size", 5, "Size of the map")
    flags.DEFINE_float("render_every", 1000, "Render the nth episode")

def get_filename():
    import config
    FLAGS = config.flags.FLAGS

    return "s-"+FLAGS.scenario+"-h-"+str(FLAGS.history_len)