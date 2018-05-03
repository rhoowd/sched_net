#!/usr/bin/env python
# coding=utf8
from __future__ import print_function
from __future__ import division
import sys
import re
import matplotlib.pyplot as plt
import logging


if __name__ == '__main__':
    num_file = len(sys.argv)-1
    filename = []
    window_size = 10000
    for i in range(1, num_file+1):
        filename.append(sys.argv[i])
        print(sys.argv[i])
    print(num_file)

    fig = plt.figure()
    for i in range(num_file):
        x = []
        y = []

        r_sum = 0
        f = open(filename[i])
        step = 0
        for line in f:

            if line.split("\t")[1].split(" ")[0] == "tr_ep_step":
                step += 1
                r_str = line.split("\t")[2]
                reward = float(r_str)

                r_sum += reward
                if step % window_size == 0:
                    x.append(step)
                    y.append(r_sum/float(window_size))
                    r_sum = 0
        plt.plot(x, y)

    plt.xlabel('Training steps', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    plt.grid(True)
    # plt.ylim(0, 2.5)
    # plt.show()

    # Save to pdf file
    plt.savefig("plot_test.pdf")
