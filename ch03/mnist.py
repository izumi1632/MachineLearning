import matplotlib.pylab as plt
import pickle
import numpy as np

import sys,os
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

from dataset.mnist import load_mnist
from PIL import Image
import utility.utility as utl
import utility.utility as sigmoid
import utility.utility as softmax

ch03_dir = os.path.dirname(os.path.abspath(__file__))
sample_weight_path = ch03_dir + "/sample_weight.pkl"


def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open(sample_weight_path,'rb') as f:
        network = pickle.load(f)
    return network

def predict (network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = utl.sigmoid(a1)
    a2 = np.dot(z1,W2) + b2 
    z2 = utl.sigmoid(a2)
    a3 = np.dot  (z2 , W3)+b3
    y = utl.softmax(a3)

    return y 

def f3_6_2():
    x,t = get_data()
    network = init_network()
    batch_size = 100
    accuracy_cnt = 0

    # for i in range(len(x)):
    #     y = predict(network,x[i])
    #     p = np.argmax(y)
    #     if p == t[i]:
    #         accuracy_cnt += 1

    for i in range(0,len(x),batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network,x_batch)
        p = np.argmax(y_batch,axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print("Accuracy:" + str(float(accuracy_cnt)/ len(x)))

f3_6_2()
print(sample_weight_path)    