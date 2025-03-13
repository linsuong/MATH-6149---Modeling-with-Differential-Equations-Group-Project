import numpy as np
import scipy
from matplotlib import pyplot as plt
import pandas as pd
import scipy
import scipy.integrate

g = 9.81
f = 0.1
alpha = 0.5
sigma = 0.5
delta = 5
V = 10
A_L = 10

shape = 'Rectangle'

def l(A):
    if shape == 'Rectangle':
        w = 10
        
        return w + (2 * A)/w

    if shape == 'Wedge':
        theta = np.pi/6
        
        return np.sqrt((8 * A)/(np.sin(theta)))
    
    if shape == 'Semi':
        theta = np.pi/3
        
        return np.sqrt((2 * A)/(theta - np.sin(theta))) * theta
    
def u_bar(A):
    return np.sqrt((g * np.sin(alpha) * A)/(f * l(A)))

"""
def A_i(A):
    return A * np.sqrt((A * g * np.sin(alpha))/f * l(A))
"""
    
def int_cond(s):
    b = 0
    norm = np.sqrt(2 * np.pi * (sigma ** 2))
    
    return (V/norm) * np.exp(-((s - b)**2) / (sigma**2)) + A_L
    
def Q(A):
    return A * u_bar(A)

def c(A_1, A_2):
    # A_1 = A_i, A_2 = A_{i-1}
    
    return (Q(A_1) - Q(A_2))/delta

A_total = []
y0 = []

for i in range()
    A_i = scipy.integrate.odeint(u_bar, y0, t)



    A_total.append(A_i)