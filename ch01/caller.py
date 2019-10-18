import sys,os
print(sys.path)
print(os.pardir)

# sys.path.append(os.pardir) ←ではダメだった。。。
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)
from dataset.mnist import load_mnist


(xtrain,ttrain),(xtest,ttest)=load_mnist(flatten=True,normalize=False)
print(xtrain.shape)
print(ttrain.shape)
print(xtest.shape)
print(ttest.shape)

# p.f3_5_3()                              