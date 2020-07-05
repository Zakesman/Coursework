# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:38:52 2020

@author: Zukisa
"""

import numpy as np

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z-- numpy array of any shape
    
    Returns:
    A-- output of sigmoid(Z), same shape as Z
    cache -- returns Z as well. Useful for backpropagation
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implements the ReLU function.
    
    Arguments:
    Z-- Output of the linear layer of any shape
        
    Returns:
    A-- post-activation parameter of the same shape as Z.
    cache-- a Python dictionary containing "Z". Stored for computing
            the backward pass in backprop efficiently
    """
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z
    
    return A, cache

def relu_backward(dA,cache):
    """
    Implement backpropagation for a single ReLU unit.
    
    Arguments:
    dA -- post-activation gradient of any shape
    cache -- where we store Z for computing backprop efficiently
        
    Returns:
    dZ -- gradient of the cost function with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA,copy=True) # converting dz to the correct object

    dZ[Z<=0] = 0 
    
    assert(dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA,cache):
    """
    Implements backpropagation for a single unit
    
    Arguments:
    dA-- post-activation gradient of any shape
    cache-- where we store Z for efficient compuation of backpropagation
        
    Returns:
    dZ -- gradient of the cost function w.r.t Z
    
    """
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = np.multiply(dA,s*(1-s))
    #dZ = np.multiply(dA,np.multiply(s,(1-s)))
    
    assert(dZ.shape == Z.shape)
    
    return dZ


    
    
    
    
    
    