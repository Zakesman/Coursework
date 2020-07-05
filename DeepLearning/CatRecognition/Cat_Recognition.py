# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 05:09:52 2019

@author: Zukisa Mbuli
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
#import skimage

#Loading the data of cat/non-cat images
train_set_x_orig,train_set_y, test_set_x_orig,test_set_y, classes = load_dataset()

index = 25
plt.imshow(train_set_x_orig[index])
print("y=" + str(train_set_y[:,index]) + ", it's a " \
                 + classes[np.squeeze(train_set_y[:,index])].decode("utf-8")\
                 + " picture.")

# DATA PREPROCESSING
#1. Understanding array dimensions
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
print("m_train:" , m_train ,"; m_test:", m_test,"; num_px:", num_px )

#Unrolling the images into vectors
train_set_x_flat = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flat = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("train_set_x_flat is of ",train_set_x_flat.shape,"dimensions")
print("test_set_x_flat is of ",test_set_x_flat.shape,"dimensions")

#2. Standardizing the image dataset. It's common to standardize image data
# by dividing each row in the dataset by 255 which is the max pixel value
train_set_x = train_set_x_flat/255
test_set_x = test_set_x_flat/255

def sigmoid(z):
    """
    This is the activation function that is used by the neurons to fit 
    a more complex function to the data.
    
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    if type(z) == np.ndarray or np.isscalar(z) == True:
        s = 1./(1+np.exp(-z))
    
    else:
        s = 1./(1+np.exp(-1*np.array(z)))
        
    return s

#Parameteter Initialization. Initialise the weights w with zeros
def initialize_with_zeros(dim):
    """
    This function initializes parameters for the network names, the weights (w) 
    and bias (b).
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
        -- num_px*numpx*3
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    
    NOTE: for image inputs, w.shape = (num_px*num_px*3,1)
    """
    w = np.zeros(shape = (dim,1),dtype=np.float64)
    b = 0
    
    assert w.shape == (dim,1)
    assert (isinstance(b,float) or isinstance(b, int))
    
    return w, b
#--------test initialize_with_zeros()
"""
dim = 2
w,b = initialize_with_zeros(dim)
print("w = ", str(w))
print("b = ", str(b))
"""    
def propagate(w,b,X,Y):
    """
    This performs the forward and backpropagation used for learning the 
    parameters.
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    """------ for testing purposes-----
    X = train_set_x
    Y = train_set_y
    b = 0
    dim = train_set_x.shape[0]
    w = w = np.zeros(shape = (dim,1),dtype=np.float64)
    """
    # Forward Propagation
    m = X.shape[1]
    activation = sigmoid(np.dot(w.T,X)+b)
    #cost  = (-1./m)*np.sum(np.dot(Y.T,np.log(activation))+np.dot((1-Y).T,np.log(1-activation)),axis = 1)
    cost = (-1. / m) * np.sum((Y*np.log(activation) + (1 - Y)*np.log(1-activation)), axis=1)
    
    # Backward Propagation
    db = (1./m)*np.sum(activation-Y,axis=1) #gradient of loss w.r.t b
    dw = (1./m)*np.dot(X,((activation-Y).T)) # gradient of loss w.r.t w
    
    assert dw.shape == w.shape
    assert db.dtype == float
    cost = np.squeeze(cost)
    assert cost.shape == ()
    
    gradients = {"dw":dw,"db":db}
    
    return gradients, cost

#-----testing propagate() 
"""    
w,b,X,Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([1,0])
gradients, cost = propagate(w,b,X,Y)
print("dw = ", gradients["dw"])
print("db = ", gradients["db"])
print("cost", cost)
"""

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    """
    The function updates parameters w and b using gradient descent by minimising
    the cost function.For parameter theta, the update rule is 
    theta_new = theta_old - alpha*del_theta, alpha is the learning rate.
            
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- if True then print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs = []
    for i in range(num_iterations):
       
       # Cost and gradient calculations
       gradients, cost = propagate(w,b,X,Y)
       
       # Retrieve grads from outut
       dw = gradients["dw"]
       db = gradients["db"]
       
       # Perform the update
       w = w-learning_rate*dw
       b = b-learning_rate*db
       
       # Record the costs every 100 iterations
       if i % 100 == 0:
           costs.append(cost)
           
       if print_cost == True and i % 100 == 0:
           #print ("Cost after iteration %i: %f" %(i, cost))
           print("The cost at iteration %s is %s" %(i,cost) )
           
    parameters = {"w":w,"b":b}
    gradients = {"dw":dw,"db":db}
    
    return parameters, gradients, costs

# -----Test the optimize() function
"""    
parameters, gradients, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(parameters["w"]))
print ("b = " + str(parameters["b"]))
print ("dw = " + str(gradients["dw"]))
print ("db = " + str(gradients["db"]))
"""

def predict(w,b,X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    #Compute A the probability vector of a cat being in the image
    A = sigmoid(np.dot(w.T,X)+b)
    
    [print(x) for x in A]
    for i in range(A.shape[1]):
        if A[0, i] >= 0.5:
            
            Y_prediction[0, i] = 1
            
        else:
            Y_prediction[0, i] = 0
            
    assert Y_prediction.shape == (1,m)
    
        
    return Y_prediction

# Test the predict() function
"""
print("predictions = " + str(predict(w,b,X)))         
"""

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000,
          learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    # Initialize parameters to zeros
    w, b = initialize_with_zeros(X_train.shape[0])
    
    # Gradient descent to obtain updates of w and b
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    
    # retrieve updated parameters from the paramaters dictionary
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict train/test set examples
    Y_prediction_train = predict(w,b,X_train)
    Y_prediction_test = predict(w,b,X_test)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
    

d = model(train_set_x, train_set_y, test_set_x, test_set_y, 
          num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    
# Example of wrongly classified image
  
index = 5
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + 
       ", you predicted that it is a \"" + 
       classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")   
"""   
    
    
# plotting the learning curve
costs = np.squeeze(d["costs"])
plt.plot(costs)
plt.ylabel("Cost")
plt.xlabel("Iterations (per hundred)")
plt.title("Learning rate = "+ str(d["learning_rate"]))
plt.show()

#---------------------- HYPERPARAMETER ANALYSIS-------------------
# We analyse the effect of different learning rate sizes on the performance
# of the classifier.
learning_rates = [0.0055, 0.00055, 0.0001]    
models = {}
for i in learning_rates:
    print("Learning rate is:" + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, 
          num_iterations = 1500, learning_rate = i, print_cost = False)
    print('\n' + "------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]),label = str(models[str(i)]["learning_rate"]))

plt.ylabel("Cost")
plt.xlabel("Iterations in hundreds")

legend = plt.legend(loc = "upper center", shadow = "True")
frame  = legend.get_frame()
frame.set_facecolor("0.90")
plt.show"""



    
    
    
    
    
    
    
    
    
    
    
    
