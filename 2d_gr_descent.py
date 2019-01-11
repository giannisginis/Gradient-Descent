#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:39:26 2019

@author: igkinis

This programm implements 2d linear regression with gradient descent for optimization without 
any machine learning libraries. Just Numpy.

"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
N,D = iris.data.shape
iris_X = np.zeros((N,D+1))
iris_X [:,0]=1
iris_X[:,1:5] = iris.data
iris_y = iris.target
#print the shape of X,y
print("the shape of x is: ", iris_X.shape)
print("the shape of y is" , iris_y.shape)


class grad:
    
    def __init__(self):
        pass
    
    def gradient_desc(self,X,y):
        # let's try gradient descent
        N,D = iris.data.shape
        costs = [] # keep track of squared error cost
        w = np.random.randn(D+1) / np.sqrt(D+1) # randomly initialize w
        learning_rate = 0.0001
        for t in range(100):
            # update w
            Yhat = iris_X.dot(w)
            delta = Yhat - iris_y
            w = w - learning_rate*iris_X.T.dot(delta)
    
    
        # find and store the cost
            mse = delta.dot(delta) / N
            costs.append(mse)
        print(Yhat.shape, w)
        return Yhat, costs, w

    def plots (self, costs, Yhat, Y, w):
        # plot the costs
        plt.title('Objective func', size = 20)
        plt.xlabel('No. of iterations', size = 20)
        plt.ylabel('Costs', size = 20)
        plt.plot(costs, label='error rate')
        plt.legend()
        plt.show()
        print("\nfinal weights:", w)
    
        # plot prediction vs target
        plt.plot(Yhat, label='prediction')
        plt.plot(Y, label='target')
        plt.legend()
        plt.show()
    
    def rsq(self, prediction, y_test):
        #
        total_data = len(prediction)
        #Average of total prediction 
        y_avg = np.sum(y_test)/total_data
        #total sum of square error
        tot_err = np.sum((y_test-y_avg)**2)
        #total sum of squared error of residuals
        res_err = 0
        x= len(prediction)
        for i in range(x):
            res_err += ((y_test[i]-prediction[i])**2)
        #
        r2 = 1 - (res_err / tot_err)
        return r2 
grad = grad()
Yhat, costs, w = grad.gradient_desc(iris_X, iris_y)
grad.plots(costs, Yhat, iris_y, w)
r2_val = grad.rsq(Yhat, iris_y)
print('R squared value', r2_val)
