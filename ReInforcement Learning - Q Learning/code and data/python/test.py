# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:05:05 2019

@author: cheth
"""

# STUDENTS: DO NOT CHANGE THIS FILE
from __future__ import division
import numpy as np
import math
from six.moves import zip
from tiles import tiles, IHT

# Utility Functions
class Error(Exception):
    pass

# Randomness is only used for reset()-ing the environment
def np_random(seed=None):
    if seed is not None and not (isinstance(seed, integer_types) and 0 <= seed):
        raise Error('Seed must be a non-negative integer or omitted, not {}'.format(seed))
    rng = np.random.RandomState()
    if seed is None:
        seed = rng.randint(2**32 - 1)
    rng.seed(seed)
    return rng, seed

x, y = np_random()

print(x)
print(y)
