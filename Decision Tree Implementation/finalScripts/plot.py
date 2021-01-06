# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:01:12 2019

@author: cheth
"""


from matplotlib import pyplot as plt

x=[0,1,2,3,4,5,6,7]
error_train=[0.4430, 0.2013, 0.1342, 0.1141, 0.1074, 0.0872, 0.0738, 0.0671]
error_test=[0.5060, 0.2169, 0.1566,0.1687, 0.2048, 0.1687, 0.1928, 0.2169]
plt.plot(x,error_train, label='Train Error')
plt.plot(x,error_test, label="Test Error")
plt.title('Train error vs Test Error')
plt.xlabel('Max Depth')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.show()