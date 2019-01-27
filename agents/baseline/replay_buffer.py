#!/usr/bin/env python
# coding=utf8
import logging
import config
from collections import deque
import random

FLAGS = config.flags.FLAGS

logger = logging.getLogger("Agent.replay")
result = logging.getLogger('Result')


class ReplayBuffer:
    def __init__(self):
        self.replay_memory_capacity = FLAGS.b_size  # capacity of experience replay memory
        self.minibatch_size = FLAGS.m_size  # size of minibatch from experience replay memory for updates
        self.replay_memory = deque(maxlen=self.replay_memory_capacity)

    def add_to_memory(self, experience):
        self.replay_memory.append(experience)

    def sample_from_memory(self):
        return random.sample(self.replay_memory, self.minibatch_size)

    def erase(self):
        self.replay_memory.popleft()
