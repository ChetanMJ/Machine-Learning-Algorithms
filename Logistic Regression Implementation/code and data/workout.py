# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 20:30:45 2019

@author: cheth
"""

import numpy as np
x = np.array([[0, 10, 20], [20, 30, 40]])
print("Original array: ")
print(x)
print("Values bigger than 10 =", x[x>10])
print("Their indices are ", np.nonzero(x > 10))