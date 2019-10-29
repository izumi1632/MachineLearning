import numpy as np

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

# ソフトマックス
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #オーバーフロー対策
    return exp_a / np.sum(exp_a)

# ２条和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 交差エントロピー誤差
# バッチ対応版
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    dlt = 1e-7
    batch_size = y.shape[0]

    return - np.sum(t * np.log(y + dlt)) / batch_size

    #one-hot表現ではない場
    # return - np.sum(t * np.log(y[np.arrange(batch_size),t]) + dlt)) / batch_size

def numerical_diff(f, x):
    h = 1e-4 #0.0001
    return (f(x+h) - f(x-h) / (2*h))

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

def gradient_descent(f, init_x, lr = 0.001, step_num =100):
    x = init_x

    for i in range(step_num):
        grad = numerical_diff(f,x)
        x -= lr * grad
    return x
