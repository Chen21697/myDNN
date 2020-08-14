#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 20:02:42 2020

@author: yuwenchen
"""
import numpy as np
from structure import *
import matplotlib.pyplot as plt


x = np.linspace(-1, 1, 200)[:, None]
y = x ** 2
learning_rate = 0.001


        
if __name__ == '__main__':
    myDNN = DNN(x, y)
    
    for i in range(300):
        myDNN.calculate()
        print(myDNN.lossFunc())
        
        test = myDNN.find_gradient()
        myDNN.updatdParameter(test, learning_rate)
            
plt.plot(x, myDNN.layer3.z)
plt.show()