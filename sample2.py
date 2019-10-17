import numpy as np
import matplotlib.pyplot as plt

X = np.array([[51,55],[14,19],[0,4]])
print(X)
print(X[0])
print(X[0,1])
for row in X:
    print(row)

X = X.flatten()
print(X[np.array([0,2,4])])