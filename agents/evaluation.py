#!/usr/bin/env python
# coding=utf8

import numpy as np
import logging
import config

FLAGS = config.flags.FLAGS
result = logging.getLogger('Result')

class Evaluation(object):

    def __init__(self):
        self.episode_cnt = 0
        self.m = dict()

    def update_value(self, m_key, m_value, m_append=None):
        if m_key in self.m:
            self.m[m_key]['value'] += m_value
            self.m[m_key]['cnt'] += 1
        else:
            self.m[m_key] = dict()
            self.m[m_key]['value'] = m_value
            self.m[m_key]['cnt'] = 1
        if m_append is None:
            result.info(m_key + "\t" + str(m_value))
        else:
            result.info(m_key + "\t" + str(m_value) + "\t" + str(m_append))

    def summarize(self, key=None):
        if key is None:
            for k in self.m:
                print "Average", k, float(self.m[k]['value'])/self.m[k]['cnt']
                result.info("summary\t" + k + "\t" + str(float(self.m[k]['value']) / self.m[k]['cnt']))

        elif key not in self.m:
            print "Wrong key"

        else:
            print "Average", key, float(self.m[key]['value']) / self.m[key]['cnt']
            result.info("summary\t" + key + "\t" + str(float(self.m[key]['value'])/self.m[key]['cnt']))
