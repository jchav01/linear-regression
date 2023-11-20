# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:07:50 2023

@author: jules
"""

import numpy as np
import matplotlib.pyplot as plt

########### GENERATING RANDOM POINTS AROUND A LINEAR CURVE ####################
np.random.seed(42)

x = np.linspace(0, 100, 50)
y_true = 3*x + 3
y = y_true + np.random.normal(scale=20, size=50)

plt.scatter(x, y)

############# LINEAR FITTING WITH GRADIENT DESCENT ############################

w_init = 7
b_init = 0


plt.plot(x, w_init*x + b_init, label='Initial Guess', color='red')

def cost(f, y):
    m = len(y)
    return 1/(2*m) * np.sum((f-y)**2)

def gradient_descent(x, w, b, y, learning_rate=0.000001):
    m = len(y)
    f = w * x + b
    dw = (1/m) * np.sum((f - y) * x)
    db = (1/m) * np.sum(f - y)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

def main(x, w, b, y, num_iter):
    
    J_hist = []
    
    for i in range(num_iter):
        f = w*x + b
        cout = cost(f, y)
        
        if i % 200 == 0:
            print("cost =", cout)
            plt.plot(x, f)
            
        w, b = gradient_descent(x, w, b, y)
        J_hist.append(cout)
        
    return w, b, J_hist

num_iter = 2000
w_final, b_final, J_hist = main(x, w_init, b_init, y, num_iter)


############ Plotting the results #############################################

plt.plot(x, w_final*x + b_final, label='Final Fit', color='green')
plt.legend()
plt.show()

it = np.linspace(0, num_iter, num_iter)
plt.plot(it, J_hist, label = "Cost with iterations")
plt.legend()
plt.show()
        
w_list = np.linspace(0, 6, 50)
b_test = 3
cout_list = []
for w_test in w_list:
    
    f_test = w_test*x + b_test
    cout = cost(f_test, y)
    cout_list.append(cout)

plt.plot(w_list, cout_list, label ="Cost function with respect to w, b is fixed")
plt.legend() 

    
    