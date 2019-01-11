#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:09:05 2019

@author: igkinis
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
iris_X = np.sort(iris.data[:,2])
iris_y = iris.target
print("the shape of x is: ", iris_X.shape)
print("the shape of y is" , iris_y.shape)

x_train, x_test, y_train, y_test = train_test_split(
                 iris_X, 
                 iris_y, 
                 test_size=0.2, 
                 random_state=42)
print("the shape of x_train is: ", x_train.shape)
print("the shape of x_test is" , x_test.shape)

class linear_regression():
    def __init__(self):
        pass
    def qme(slope,inter,x_test,y_test):
        preds = []
        rmse=0
        d=x_test.shape
        pred = 0
        for i in range(d[0]):
            pred+=(slope*x_test[i]+inter)
            rmse+=(y_test[i]-(((slope*x_test[i])+inter)**2))
            preds.append(slope*x_test[i]+inter)
        error = rmse / len(x_test)
        print(len(preds), x_test.shape)
        print('Error value of the model', error)
        return error,preds
    
    def gradient_desc(slope,inter, x_train,y_train, l_rate
                      ,iter_val):
        for i in range(iter_val):
            int_slope= 0 #gradients
            int_inter= 0 #gradients
            n_pt = float(len(x_train))
            d=x_train.shape
            for i in range(d[0]):
                int_inter += - (2/n_pt) * (y_train[i] - ((slope * x_train[i]) + inter))
                int_slope += - (2/n_pt) * x_train[i] * (y_train[i] - ((slope * x_train[i]) + inter))
            
            final_slope = slope - (l_rate*int_slope)
            final_inter = inter - (l_rate*int_inter)
        print('Slope of the model',final_slope)
        print('Intercept of the model', final_inter)
        return final_slope, final_inter
            
    
    
    def rsq(prediction, y_test):
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
        print('R squared value', r2)
        return r2 
    
    def plots (Yhat, Y):
       
        # plot prediction vs target
        plt.plot(Yhat, label='prediction')
        plt.plot(Y, label='target')
        plt.legend()
        plt.show()


#defining slope and intercept value as 0 
learning_rate = float(input("Give me the l rate: "))
start_slope = float(input("Give me the start slope: "))
start_intercept = float(input ("Give me the start intercept: "))
iteration = int(input("Give me the iterations: "))
grad_slope , grad_inter = linear_regression.gradient_desc(start_slope,
                                        start_intercept,
                                        x_train, y_train, 
                                        learning_rate,
                                        iteration)

final_e_value, prediction = linear_regression.qme(
        grad_slope, grad_inter, 
        x_test, y_test)
linear_regression.plots(prediction,y_test)
linear_regression.rsq(prediction, y_test)


