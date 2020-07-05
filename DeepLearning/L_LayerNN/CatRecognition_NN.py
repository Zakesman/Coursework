# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:41:36 2020

@author: Zukisa
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from dnn_utils import *
from lr_utils import *

plt.rcParams['figure.figsize'] = (5.0,4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y,classes = load_dataset()    
#print("Dimensions of training dataset are: ",train_x_orig.shape)
#print("Dimensions of training label dataset are: ",train_y.shape)
#print("Dimensions of classes are: ",classes.shape)
#print("classes: ",classes)

# 1. DATA PROCESSING AND PREPARATION
# EXPPLORING THE DATASET
# See an example image in the dataset
index = 10
plt.imshow(train_x_orig[index])
print("y= " +str(train_y[0,index]) +", therefore this is a "+ classes[train_y[0,index]].decode("utf-8")+" image.")

m_train = train_x_orig.shape[0]  # number of training examples
num_px = train_x_orig.shape[1] # number of pixels
m_test = test_x_orig.shape[0] # number of test examples

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T # the -1
# makes reshape() flatten the remaining dimensions.
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T

# standardize
train_x = train_x_flatten/255
test_x = test_x_flatten/255

print("train_x shape: ",train_x.shape)
print("test_x shape: ",test_x.shape)

# CONSTRUCT LEARNING MODEL

n_x = num_px*num_px*3 # result is 12288, size of input layer
n_h = 7     # number of neurons in the hidden layers = size of hidden layer
n_y = 1     # size of output layer
layer_dims = (n_x,n_h,n_y)

def two_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False):
    """
    Implements a two-layer neural network
    
    Arguments:
    X-- input data, shape (n_x,number of examples)
    Y-- labels vector (1,number of examples), 0 if non-cat and 1 if cat.
    layers_dims -- dimensions of the network layers (n_x,n_h,n_y)
    learning rate -- learning rate of the gradient descent algorithm
    num_iterations -- number of iterations of the inner loop
    print_cost -- if True prints the cost function output at every 100 iterations
    
    Returns:
    parameters -- dicionary of parameters W1,W2,b1,b2
        
    """
    np.random.seed(1)
    """
    grads = {}
    costs = []      # to keep track of the cost value for printing whenr required
    m = X.shape[1]   # number of examples
    (n_x,n_h,n_y) = layers_dims
    
    #Initialise parameters
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for iteration in range(0,num_iterations):
        A1,cache1 = linear_activation_forward(X,W1,b1,"relu")
        A2,cache2 = linear_activation_forward(A1,W2,b2,"sigmoid")
        
        cost = compute_cost(A2,Y)
        
        dA2 = np.divide(1-Y,1-A2)-np.divide(Y,A2)
        
        dA1,dW2,db2 = linear_activation_backward(dA2,cache2,"sigmoid")
        dA0,dW1,db1 = linear_activation_backward(dA1,cache1,"relu")
        
        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2
        
        parameters = update_parameters(parameters,grads,learning_rate)
        
        if print_cost and iteration%100==0:
            print("Cost after iteration " + str(iteration)+" is: "+ str(cost))
            costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    """
    
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    ### START CODE HERE ### (≈ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
        ### END CODE HERE ###
        
        # Compute cost
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(A2, Y)
        ### END CODE HERE ###
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
        ### END CODE HERE ###
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        ### START CODE HERE ### (approx. 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    
    
    return parameters

parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)


