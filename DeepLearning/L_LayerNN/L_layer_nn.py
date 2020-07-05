# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 05:25:49 2020

@author: Zukisa
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from  testCases import *
from dnn_utils import sigmoid, sigmoid_backward,relu,relu_backward

np.random.seed(1)
#Initialize parameters for 2 layer network

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# Test initialize_parameters()
"""    
parameters = initialize_parameters(2,2,1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
"""

# Initialize parameters for L-layer network
def initialize_parameters_deep(layer_dims):
    """
    Randomly initialize parameters for an L-layer NN
    
    Arguments:
    layer_dims-- vector (Python list) of the number of nodes in each layer of NN
    
    Returns:
    parameters -- Python dictionary containing parameters W1,b1,...,WL,bL
                WL -- weight matrix of shape (layer_dims[L],layer_dims[L-1])
                bL -- bias vector of shape (layer_dims[L],1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) #total number of layers in network. Includes input layer
    
    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
        
        assert(parameters["W" + str(l)].shape == (layer_dims[l],layer_dims[l-1]))
        assert(parameters["b" + str(l)].shape == (layer_dims[l],1))
        
    return parameters
   
# Test initialize_parameters_deep()
"""
parameters = initialize_parameters_deep([5,4,3,1])  

print("W1 = ", parameters["W1"])
print("b1 = ", parameters["b1"])
print("W2 = ", parameters["W2"])
print("b2 = ", parameters["b2"])
print("W3 = ", parameters["W3"])
print("b3 = ", parameters["b3"])
"""

def linear_forward(A,W,b):
    """
    Computes the linear part of a layer during forward propagation
    
    Arguments:
    A -- activations from preceding layer or input data( for A^[0])
         has shape:(layer_dims[l-1],no of training examples)
    W -- weights matrix of shape (layer_dims[l],layer_dims[l-1])
    b -- bias vector of shape (layer_dims[l],1)
    
    Returns:
    Z -- input of activation function a.k.a pre-activation parameter
    cache -- Python dictionary containing A,W and b; stored for efficient 
             computing of backpropagation.
    
    """
    #m = A.shape[1]
    Z = np.dot(W,A) + b
    
    assert(Z.shape == (W.shape[0],A.shape[1]))
    
    #cache = {"A":A,"W":W,"b":b}
    cache = (A,W,b)
    
    return Z, cache

# Test linear_forward()
"""
A, W, b = linear_forward_test_case()

Z, linear_cache = linear_forward(A, W, b)
print("Z = " + str(Z))
"""

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    if activation == "sigmoid":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
        
    elif activation == "relu":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)
        
    assert(A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)
        
    
    return A,cache
    
#Test linear_activation_forward()
"""
A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))
print("Sigmoid cache:", linear_activation_cache)

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))    
""" 

def L_model_forward(X,parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    caches = []
    A = X
    L = len(parameters)//2
    
    # Compute L-1 layers for ReLU. Add cache to caches list
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
    
        caches.append(cache)
        
    # Compute single sigmoid output layer. AL denotes A^[L] = sigmoid(Z)
    
    AL,cache = linear_activation_forward(A,parameters["W"+str(l)],parameters["b"+str(l)],"sigmoid")
    
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    
    return AL,caches

#Test L_model_forward()
"""
X,parameters = L_model_forward_test_case()
AL,caches = L_model_forward(X,parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))
"""

def compute_cost(AL,Y):
    """
    Implement the computation of the cost function
    
    Arguments:
    AL -- probability vector corresponding to label predictions, shape (1,no of examples)
    Y -- vector of labels, shape (1,no of examples), (values e.g.1s and 0s for binary classification)
    
    Return:
    cost -- cross-entropy cost, scalar
    """
    m = Y.shape[1]
    
    cost = -(1/m)*np.sum(np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL)))
      
    cost = np.squeeze(cost) #ensure correct dirmensions of cost e.g. turns [[17]] into 17.
    
    assert(cost.shape == ())
    
    return cost
    
# Test compute_cost(), check why the result is different from example test case

"""
Y, AL = compute_cost_test_case()

print("cost = " + str(compute_cost(AL, Y)))
"""
def linear_backward(dZ,cache):
    """
    Implement the linear part of backpropagation for a single layer (layer L)
    
    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (current layer l)
    cache -- tuple of values (A_prev,W,b) coming from forward propagration in current layer
    
    Returns:
    dA_prev -- gradient of cost w.r.t activation of previous layer, shape(dA_prev)=shape(A_prev)
    dW -- gradient of cost w.r.t W (current layer l), shape(dW) = shape(W)
    db -- gradient of cost w.r.t b (current layer l), shape(db) = shape(b)
       
    """
    
    
    A_prev,W,b = cache
    m = A_prev.shape[1]
    
    dW = (1/m)*np.dot(dZ,A_prev.T)
    db = (1/m)*np.squeeze(np.sum(dZ,axis=1,keepdims=True))
    dA_prev = np.dot(W.T,dZ)
    
    assert(dW.shape == W.shape)
    assert(dA_prev.shape == A_prev.shape)
    assert(isinstance(db,float))
    
    return dA_prev,dW,db
    
#Test linear_backward() function
"""  
dZ,linear_cache = linear_backward_test_case()
dA_prev,dW,db = linear_backward(dZ,linear_cache)

print("dA_prev: ",dA_prev)
print("dW: ",dW)
print("db: ",db)
"""
def linear_activation_backward(dA,cache,activation):
    
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache,activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        
    dA_prev,dW,db = linear_backward(dZ,linear_cache)
        
    
    return dA_prev,dW,db
    
# Test linear_activation_backward()   
"""    
AL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
"""

def L_model_backward(AL,Y,cache):
    """
    Implement backprop for (LINEAR->ReLU)*L-1 -> LINEAR->SIGMOID. 
    
    Arguments:
    AL-- probability vector output of forward propgation process (L_model_forward())
    Y-- labels vector (1 if cat, 0 if non-cat)
    cache -- 
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...    
               
    """
    grads = {}
    
    L = len(cache) # number of layers
    m = AL.shape[1]  # number of training examples
    Y = Y.reshape(AL.shape)   
    
    # Initialise backpropagation
    dAL = np.divide(1-Y,1-AL) - np.divide(Y,AL)
    
    """
    Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches".
    Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    """
    current_cache = cache[-1]
    grads["dA"+str(L)],grads["dW"+str(L)],grads["db"+str(L)]  =linear_backward(
            sigmoid_backward(dAL,current_cache[1]),current_cache[0])
    
    
    for l in reversed(range(L-1)):
        current_cache = cache[l]
        dA_prev_temp,dW_temp,db_temp = linear_backward(sigmoid_backward(
                dAL,current_cache[1]),
        current_cache[0])
        
        grads["dA"+str(l+1)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_prev_temp
        grads["db"+str(l+1)] = db_prev_temp      
                
    
    return grads

# Test L_model_backwards()
"""    
AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dA1 = "+ str(grads["dA1"]))
"""

def update_parameters(parameters,grads,learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    """
    L = len(parameters)//2
    alpha = learning_rate
    #update rule for each parameter
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)]-alpha*grads["W"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)]-alpha*grads["b"+str(l+1)]
    """
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (≈ 3 lines of code)
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]    
    
    return parameters
  
# Test parameter_update()
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = " + str(parameters["W1"]))
print ("b1 = " + str(parameters["b1"]))
print ("W2 = " + str(parameters["W2"]))
print ("b2 = " + str(parameters["b2"]))
#print ("W3 = " + str(parameters["W3"]))
#print ("b3 = " + str(parameters["b3"]))

