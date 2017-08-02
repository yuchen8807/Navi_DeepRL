#!/usr/bin/env python
# coding=utf-8

import rospy
import numpy as np

if __name__ == '__main__':
    file = open('results/memory.csv', 'r')
    samples = eval(file.read())
    print('actions', np.array(samples['actions']))
    print('image_array', np.array(samples['states']))
