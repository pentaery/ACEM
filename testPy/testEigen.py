import numpy as np
from scipy.linalg import eig

# 定义矩阵 A 和 B
A = np.array([[3,-1,0],[-1,2,-1],[0,-1,2]])
B = np.array([[2.5,0,0],[0,1,0],[0,0,1.5]])

# 求解广义特征值问题
eigenvalues, eigenvectors = eig(A, B)

# 输出特征值
print("特征值:")
print(eigenvalues)

# 输出特征向量
print("特征向量:")
print(eigenvectors)

result = np.dot(eigenvectors[:,2].T, np.dot(B, eigenvectors[:,1]))
print(result)