import matplotlib.pylab as plt
import numpy as np
import sys,os
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from dataset.mnist import load_mnist
from PIL import Image

def print_border():
    print('------------------------------')

def __init__(self):
    self.x_train,self.t_train,self.x_test,self.t_test

def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(x*w)+b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def OR(x1,x2):
    x =np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1,x2):
    s1 =NAND(x1, x2)
    s2 =OR(x1, x2)
    y =AND(s1, s2)
    return y

def step_function(x):
    return np.array(x>0,dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def identity_function(x):
    return x


# print(AND(0,0))
# print(NAND(1,0))
# print(OR(0,1))
# print(AND(1,1))
# print(XOR(0,0))
# print(XOR(0,1))
# print(XOR(1,0))
# print(XOR(1,1))

def print_functions():
    print(step_function(np.array([1,2,0,4,-1])))
    print(sigmoid(np.array([1,2,0,4,-1])))
    x = np.arange(-5,5,0.1)
    y = step_function(x)
    y1 = sigmoid(x)
    y2 = relu(x)
    plt.plot(x,y)
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.ylim(-1,5)
    plt.show()

def f3_3_1():
    A = np.array([1,2,3,4])
    print(A)
    print(np.ndim(A))
    print(A.shape)
    print(A.shape[0])
    print('------------')
    B = np.array([[1,2],[3,4],[5,6]])
    print(B)
    print(np.ndim(B))
    print(B.shape)

def f3_3_2():
    A= np.array([[1,2],[3,4]])
    print(A.shape)
    B=np.array([[5,6],[7,8]])
    print(B.shape)
    print(np.dot(A,B))
    
    print_border()

    A = np.array([[1,2,3],[4,5,6]])
    print(A)
    B = np.array([[1,2],[3,4],[5,6]])
    print(B)
    print(np.dot(A,B))

    C = np.array([7,8])
    print(np.dot(np.array([[1,2],[3,4],[5,6]]),C))

def f3_4_2():
    X = np.array([1,0.5])
    W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    B1 = np.array([0.1,0.2,0.3])

    A1 = np.dot(X,W1)+B1
    print(A1)
    Z1 = sigmoid(A1)
    print(Z1)

    W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    B2 = np.array([0.1,0.2])

    A2 = np.dot(Z1,W2)+B2 
    Z2 = sigmoid(A2)

    W3 = np.array([[00.1,0.3],[0.2,0.4]])
    B3 = np.array([0.1,0.2])

    A3 = np.dot(Z2,W3) + B3
    Y = identity_function(A3)
    print(Y)

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']= np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    
    return network

def forward(network, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    Z1 = sigmoid(a1)
    a2 = np.dot(Z1,W2) + b2
    Z2 = sigmoid(a2)
    a3 = np.dot(Z2,W3) + b3
    y = identity_function(a3)

    return y

def f3_4_3():
    network = init_network()
    x = np.array([1.0,0.5])
    y = forward(network,x)
    print(y)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #オーバーフロー対策
    return exp_a / np.sum(exp_a)

def f3_5_3():
    a = np.array([1,2,3])
    print(softmax(a))
    print(np.sum(softmax(a)))

def f3_6_1():
    (x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=False)
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)

    img = x_train[0]
    label = t_train[0]
    print(label)
    
    print(img.shape)
    
    img = img.reshape(28,28)
    print(img.shape)

    img_show(img)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(flatten=True,normalize=True,one_hot_label=False)
    return x_test,t_test
