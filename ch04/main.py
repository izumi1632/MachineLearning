import matplotlib.pylab as plt
import pickle
import numpy as np

import sys,os
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

from dataset.mnist import load_mnist
from PIL import Image
import utility.utility as utl

# ２条和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    dlt = 1e-7
    return - np.sum(t * np.log(y + dlt))

def ex1():
    t = [0,0,1,0,0,0,0,0,0,0]
    y = [0.1,0.005,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
    s = mean_squared_error(np.array(y),np.array(t))
    print(s)

    y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
    s = mean_squared_error(np.array(y),np.array(t))
    print(s)

def ex2():
    t = [0,0,1,0,0,0,0,0,0,0]
    y = [0.1,0.005,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
    s = cross_entropy_error(np.array(y),np.array(t))
    print(s)

    y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
    s = cross_entropy_error(np.array(y),np.array(t))
    print(s)

def function1(x):
    return 0.01*x**2 + 0.1*x

def f4_3_2():
    x = np.arange(0.0,20.0,0.1)
    y = function1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.show()
    
    # np配列想定
def function2(x):
    return np.sum(x**2)

def f4_3_3():
    x1 = [1,2,3,4,5]
    x1= np.array([x1,x1])
    x = np.arange(0.0,20.0,0.1)
    plt.plot(x,x,function2(x1))
    plt.show()

# ex1()b
# ex2()
# f4_3_2()
f4_3_3()