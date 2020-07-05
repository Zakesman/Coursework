# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 05:36:09 2020

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

""" Load and visualise the dataset to be worked on. It is a flower consisting of
two sets of coloured points. The aim is to classify the two sets."""

X,Y = load_planar_dataset()

#plt.scatter(X[0,:], X[1,:], s = 40,c = Y, cmap = plt.cm.Spectral);
plt.scatter(X[0, :], X[1, :], c=np.reshape(Y,-1), s=40, cmap=plt.cm.Spectral);

# Making sense of the dataset
# Calculating the number of training examples

m = X.shape[1]
print("There are %s training examples" %(m))
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)

# Test the performance of logistic regression on this dataset

suppVec = sklearn.svm.SVC()
suppVec.fit(X.T,Y.T)

# Plot the flower dataset with the classification output
model = lambda x: suppVec.predict(x)
plot_decision_boundary(model,X,Y)
plt.title("Support Vector Machine")

svm_predictions = suppVec.predict(X.T)

print ('Accuracy of the suppport vector machine: %d ' % float((np.dot(Y,svm_predictions) + np.dot(1-Y,1-svm_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")
# Accuracy = (TP+TN)/(TP+TN+FP+FN),
# where TN= true negative, TP=  true postive, FP=false positive, FN = false negative
