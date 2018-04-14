#!/usr/bin/env python
# coding=utf8
import logging
import envs.make_env as make_env
import agents
import config
import time

FLAGS = config.flags.FLAGS

if __name__ == '__main__':

    start_time = time.time()

    # === Logging setup === #
    logger_env = logging.getLogger('GridMARL')
    logger_agent = logging.getLogger('Agent')

    # === Program start === #
    # Load environment
    env = make_env.make_env(FLAGS.scenario)
    logger_env.info('GridMARL Start with %d predator(s) and %d prey(s)', FLAGS.n_predator, FLAGS.n_prey)

    # Load trainer
    logger_agent.info("Agent")
    trainer = agents.load(FLAGS.agent+"/trainer.py").Trainer(env)

    print FLAGS.agent, config.file_name

    # start learning
    if FLAGS.train == True:
        trainer.learn()
    else:
        trainer.test()

    print "TRAINING TIME (sec)", time.time() - start_time
    print "exit"
