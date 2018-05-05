#!/usr/bin/env python
# coding=utf8

import sys
import re
import matplotlib.pyplot as plt
import logging
import numpy as np


class DataTable(object):
    def __init__(self, n_result):
        self._n_result = n_result
        self._n_row = 1
        self._data_array = [[0]*n_result]
        self.average = list()
        self.confidence = list()
        self.z = 1.96  # it is for 95%

    def print_array(self):
        print self._data_array

    def insert(self, seed, step, value):
        column = seed
        row = step
        if column >= self._n_result:
            return 0
        while self._n_row <= row:
            self._extend_array()

        self._data_array[row][column] = value

        return 0

    def _extend_array(self):
        self._data_array.append([0]*self._n_result)
        self._n_row += 1
        return self._n_row

    def calculate(self):
        for i in range(self._n_row):
            self.average.append(np.mean(self._data_array[i]))
            std = np.std(self._data_array[i])
            confidence = self.z * std / np.sqrt(self._n_result)
            self.confidence.append(confidence)

    def get_result(self):
        for i in range(self._n_row):
            print self.average[i], self.average[i]-self.confidence[i], 2*self.confidence[i], i


if __name__ == '__main__':

    num_file = len(sys.argv)-1
    filename = []
    window_size = 1
    for i in range(1, num_file+1):
        filename.append(sys.argv[i])

    d_table = DataTable(num_file)

    fig = plt.figure()
    for i in range(num_file):

        x = []
        y = []
        cnt = 0
        cnt_avg = 0
        f = open(filename[i])
        r_sum = 0
        for line in f:
            if line.split("\t")[1].split(" ")[0] == "test_result":

                # step = line.split("\t")[2]
                # r_str = line.split("\t")[3]
                # reward = float(r_str.split(',')[0][1:])
                try:
                    step = line.split("\t")[3]
                    r_str = line.split("\t")[2]
                    reward = float(r_str)
                    r_sum += reward
                    cnt += 1
                    if cnt % window_size == 0:
                        x.append(step)
                        y.append(r_sum/float(window_size))
                        d_table.insert(i, cnt_avg, r_sum / float(window_size))
                        cnt_avg += 1
                        r_sum = 0
                except:
                    print line

        plt.plot(x, y)

    d_table.calculate()
    d_table.get_result()
