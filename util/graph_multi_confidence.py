#!/usr/bin/env python
# coding=utf8

import sys
import re
import matplotlib.pyplot as plt
import logging
import numpy as np


if __name__ == '__main__':

    num_file = len(sys.argv)-1
    filename = []
    y_lim = 200

    for i in range(1, num_file+1):
        filename.append(sys.argv[i])

    leg_list = []
    for i in range(num_file):

        x = []
        y = []
        lb = []
        ub = []
        cnt = 0
        cnt_avg = 0
        f = open(filename[i])
        r_sum = 0

        for line in f:

            try:
                line_data = line.split(" ")
                x.append(int(line_data[3]))
                y.append(float(line_data[0]))
                lb.append(float(line_data[1]))
                ub.append(float(line_data[1])+float(line_data[2]))

            except:
                a = 1

        leg, = plt.plot(x, y, label=filename[i].split('/')[0])
        plt.fill_between(x, lb, ub, alpha=0.4)
        leg_list.append(leg)

    plt.legend(handles=leg_list)
    plt.xlabel('Training step', fontsize=16)
    plt.ylabel('Step to capture', fontsize=16)
    plt.grid(True)
    plt.ylim(0, y_lim)
    plt.xlim(0, 22)
    # plt.show()

    plt.savefig("plot_conf_"+str(y_lim)+".pdf")