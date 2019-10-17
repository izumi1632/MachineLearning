import matplotlib.pylab as plt
import numpy as np

def print_border():
    print('------------------------------')


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