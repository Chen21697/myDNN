#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:58:18 2020

@author: yuwenchen
"""

import numpy as np
np.random.seed(1)
#%%
class layer():
    def __init__(self, inputlen, neuron, activation, isOutput=None, isInput=None):
        
        self.isOutput = False
        self.isInput = False
        
        if isOutput:
            self.isOutput = True
        if isInput:
            self.isInput = True
        
        self.activation = activation
        self.neuron = neuron
        
        self.w = np.random.uniform(0, 1, (inputlen, neuron))
        self.b = np.full((1, neuron), 0.1)
            
            
#%%
class DNN():
    def __init__ (self, inputD, target):
        
        self.inputD = inputD
        self.target = target 
        
        self.layer1 = layer(1, 10, "tanh", isInput=True)
        self.layer2 = layer(10, 10, "tanh")
        self.layer3 = layer(10, 1, "RMS", isOutput=True)
        
    def calculate(self):
        self.layer1.z = self.inputD @ self.layer1.w + self.layer1.b
        if self.layer1.activation is "tanh":
            self.layer1.a = tanh(self.layer1.z)
            
        self.layer2.z = self.layer1.a @ self.layer2.w + self.layer2.b
        if self.layer2.activation is "tanh":
            self.layer2.a = tanh(self.layer2.z)
            
        self.layer3.z = self.layer2.a @ self.layer3.w + self.layer3.b
        
    def lossFunc(self):
        loss = np.sum((self.layer3.z - self.target)**2)/2
        return loss
    
    def find_gradient(self):
        l3_backwardPass = derivative_RMS(self.target, self.layer3.z)
        
        
        l3_forwardPass = self.layer2.a.T
        l3_w_grad = l3_forwardPass @ l3_backwardPass
        l3_b_grad = np.sum(l3_backwardPass, axis=0, keepdims=True) # add gradient of all data
        
        l2_backwardPass = l3_backwardPass @ self.layer3.w.T * derivative_tanh(self.layer2.z)
        l2_forwardPass = self.layer1.a.T
        l2_w_grad = l2_forwardPass @ l2_backwardPass
        l2_b_grad = np.sum(l2_backwardPass, axis=0, keepdims=True)
        
        l1_backwardPass = l2_backwardPass @ self.layer2.w.T * derivative_tanh(self.layer1.z)
        l1_forwardPass = self.inputD.T
        l1_w_grad = l1_forwardPass @ l1_backwardPass
        l1_b_grad = np.sum(l1_backwardPass, axis=0, keepdims=True)
        return [l1_w_grad, l2_w_grad, l3_w_grad, l1_b_grad, l2_b_grad, l3_b_grad]
    
    
    def updatdParameter(self, grads, lr):
        originalP = [self.layer1.w, self.layer2.w, self.layer3.w, self.layer1.b, self.layer2.b, self.layer3.b]        
        new_parameter = []
        
        for (param, gradient) in zip(originalP, grads):    
            param = param - lr * gradient
            new_parameter.append(param)
            
        self.layer1.w = new_parameter[0]
        self.layer2.w = new_parameter[1]
        self.layer3.w = new_parameter[2]
        self.layer1.b = new_parameter[3]
        self.layer2.b = new_parameter[4]
        self.layer3.b = new_parameter[5]
#%%
def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1 - np.power(tanh(x), 2)

def RMS(ins, target):
    return np.sum((target - ins)**2)/2

def derivative_RMS(y_head, y):
    return y - y_head 

#%%