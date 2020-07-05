# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 20:30:51 2019

@author: Zukisa
"""
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
from testCases import layer_sizes_test_case
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset

np.random.seed(1)
# CREATE FUNCTIONS REQUIRED FOR NN

def layer_sizes(X,Y):
    """
    Arguments:
    X-- input dataset of shape (# of features, # of examples)
    Y-- label vector of shape (label, # of examples)
    
    Returns:
    n_x -- the size of the input layer, extracted from the array X.
    n_h -- size of the hidden layer, hardcoded.
    n_y -- size of the output layer, extracted from the array Y.
    
    """
    n_x = X.shape[0]
    n_h = 4   # creating a neural network with 4 nodes in the hidden layer
    n_y = Y.shape[0]
    
    return n_x, n_h, n_y

# test layer_sizes()
"""    
X_assess, Y_assess = layer_sizes_test_case()
n_x,n_h,n_y = layer_sizes(X_assess, Y_assess)
print("n_x is", n_x)
print("n_h is", n_h)
print("n_y is", n_y)
"""

def initialize_parameters(n_x,n_h,n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2)
    
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    
    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y,1))
    
    parameters = {"W1":W1, "b1":b1,"W2":W2,"b2":b2}
    
    return parameters

# Test initialize_parameters()
"""
n_x, n_h,n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x,n_h,n_y)
print("Paramters are:")
print("W1:", parameters["W1"])
print("b1:", parameters["b1"])
print("W2:", parameters["W2"])
print("b2:", parameters["b2"])
"""
def forward_propagation(X,parameters):
    """
    Argument:
    X -- the input matrix of size (n_x,m), from which to extract features
    parameters -- contains the initialised weights (W) and bias terms from
                  the function initialize_parameters()
               -- stored in a python dictionary
                  
        Returns:
        Y -- A2 the sigmoid output of the second activation
          -- takes on 1 if y_predicted > 0.5 and 0 otherwise
        cache -- a dictionary containing z1, z2, a1 and a2
        
        """
        # Retrieve each parameter from the dictionary parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
        
    # Implement forward propagation to compute a2
    z1 = np.dot(W1,X)+b1
    activation1 = np.tanh(z1)
    #print("shape[a1]:", activation1.shape)
    z2 = np.dot(W2,activation1) + b2
    activation2 = sigmoid(z2)
    #print("shape[a2]:", activation2.shape)
    assert(activation2.shape == (1,X.shape[1]))
        
    cache = {"z1":z1,"A1":activation1,"z2":z2,"A2":activation2}        
        
    return activation2, cache
    
def compute_cost(activation2,Y):
    """
    The function calculates the cross entropy cost
        
    Arguments:
    activation2 -- output from the forward propagation process with shape (1,m)
    Y -- the labes vector with shape (1,m)
        
    Returns:
    cost -- scalar value from the cross-entropy cost function
    """
    assert(activation2.shape == (1,Y.shape[1]))
    m = Y.shape[1]
    cost_term1 = np.dot(Y,np.log(activation2).T)
    cost_term2 = np.dot((1-Y),np.log(1-activation2.T))
    cost = -(1/m)*sum(cost_term1+cost_term2)
        
    cost = np.squeeze(cost) #ensures that the cost is the expected dimension
    assert(cost.dtype == float) # to ensure that cost is a scalar value
        
    return cost
    """
    # Test compute cost
    a2,Y_assess,parameters = compute_cost_test_case()
    my_cost = compute_cost(a2,Y_assess)
    print("cost = ", my_cost)
    """
    
def back_propagation(parameters,cache,X,Y):
        
    """
    Input:
    parameters -- Python dictionary containing paramaters
    cache -- Python dictionary containing "a1","a2","z1","z2"
    X -- input data of shape (2, no of examples)
    Y -- true labels vector of shape (1, no of examples)
        
    Output:
    grads -- dictionary containing gradients w.r.t each parameter        
        
    """
    m = X.shape[1]  # getting the number of training examples
        
    #Retrieve W1 and W2 from 'parameters' dictionary
    #W1 = parameters["W1"]
    W2 = parameters["W2"]
        
    # Retrieve A1 and A2 from cache
    A1 = cache["A1"]
    A2 = cache["A2"]
        
    # backpropagation: calculate dW1, db1,dW2,db2
    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2,axis = 1,keepdims = True)
    dZ1 = np.multiply(np.dot(W2.T,dZ2),1-np.power(A1,2))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1,axis = 1,keepdims = True)
        
    grads = {"dW1":dW1,"db1":db1,"dW2":dW2,"db2":db2}
        
    return grads

#Test back_propgation
"""
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = back_propagation(parameters,cache,X_assess,Y_assess)

print("dW1 = ", grads["dW1"])
print("db1 = ", grads["db1"])
print("dW2 = ", grads["dW2"])
print("db2 = ", grads["db2"])
"""

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using gradient descent
    
    Inputs:
    parameters-- Python dictionary containing parameters W1,b1,W2,b2
    grads -- Python dictionary containing gradients for gradient descent
    learning_rate -- learning rate or step-size in gradient descent algorithm
    
    Outputs:
    parameters -- Python dictionary containing updated parameters
        
    """
    # Retrieve parameters from parameters dictionary
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Retrieve gradients from grads dictionary
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Update parameters with gradient descent rule
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    # Store updated values in parameters dictionary again
    parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    
    return parameters

# Test update_parameters()
"""
parameters, grads = update_parameters_test_case()
parms = update_parameters(parameters,grads)

print("W1 = ", parms["W1"])
print("b1 = ", parms["b1"])
print("W2 = ", parms["W2"])
print("b2 = ", parms["b2"])
"""

def nn_model(X,Y,n_h,num_iterations,print_cost):
    """
    Inputs:
    X-- dataset of shape (2,no of examples), there are 2 features in this set
    Y-- vector of labes data of shape (1,number of examples)
    n_h -- size of hidden layer i.e. no of nodes in hidden layer
    num_iterations -- num of gradient descent loop iterations
    print_cost -- takes on value of True or False. Determines whether cost 
                  function value should be printed every 1000 iterations
        
    Outputs:
    parameters -- Python dictionary of updated/learnt parameters. These will
                  be used for prediction
    """
    np.random.seed(3)
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    for i in range(0,num_iterations):
        A2,cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y)
        grads = back_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads)
        
        if print_cost and i % 1000 == 0:
            print("Cost of iteration %i: %f" %(i,cost))
        
    
    return parameters

#Test nn_model()
"""
X_assess,Y_assess = nn_model_test_case()
parameters = nn_model(X_assess,Y_assess,4,10000,False)
print("W1 = ", parameters["W1"])
print("b1 = ", parameters["b1"])
print("W2 = ", parameters["W2"])
print("b2 = ", parameters["b2"])
"""

def predict(parameters,X):
    """
    Inputs:
    parameters-- parameters learnt by the model
    X-- input data of shape (n_x,no of examples), n_x = 2 in this case
            
    Outputs:
    predictions-- vector of model predictions (red:0/blue:1)
            
    """
    A2,cache = forward_propagation(X,parameters)
    predictions = np.round(A2)
    
    return predictions

# Test predict()
"""
parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
print("predictions mean = " + str(np.mean(predictions)))
"""
# Build a 3 layer neural network model with n_h=4 nodes in the hidden layer
parameters = nn_model(X,Y,4,10000,True)
plot_decision_boundary(lambda x: predict(parameters,x.T),X,Y)
plt.title("Decision Boundary for NN with " + str(4) + " nodes")

"""
Plot accuracy
Accuracy = (TP+TN)/(TP+TN+FP+FN)
"""
predictions = predict(parameters,X)
print("Accuracy: %d " % float((np.dot(Y,predictions.T)+ np.dot(1-Y,1-predictions.T))/float(Y.size)* 100 )+"%")
 


# Testing different layer sizes

plt.figure(figsize=(16,32))
hidden_layer_sizes = [1,2,3,4,5,20,50]   
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5,2,i+1)
    plt.title("Hidden layer of size %d " % n_h)
    parameters = nn_model(X,Y,n_h,num_iterations=5000,print_cost=False)
    plot_decision_boundary(lambda x: predict(parameters,x.T),X,Y)
    predictions = predict(parameters,X)
    accuracy = float((np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print("Accuracy of {} hidden units: {} %" .format(n_h,accuracy))



      
        


