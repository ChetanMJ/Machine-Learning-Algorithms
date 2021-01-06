# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 02:06:54 2019

@author: cheth
"""

from matplotlib import pyplot as plt
import math
log_likely_train = [-122.5416, -110.316, -101.0720, -95.4374]
log_likely_test = [-131.6493, -116.0917, -105.3082, -98.5283]
epochs = [math.log10(10), math.log10(100), math.log10(1000), math.log10(10000)]
plt.plot(epochs,log_likely_train, label = "Train data")
plt.plot(epochs,log_likely_test, label = "Test data")
plt.title('Train/Test - Log Likelihood')
plt.xlabel('No. of Training Sequence(log base 10 scale)')
plt.ylabel('log Likely')
plt.legend(loc='bottom right')
plt.show()