import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

def is_positive_definite(csr_matrix):

    
    # 将CSR矩阵转换为稠密矩阵
    dense_matrix = csr_matrix.toarray()
    
    # 检查矩阵是否对称
    if not np.allclose(dense_matrix, dense_matrix.T):
        return False
    
    # 尝试进行LU分解
    try:
        splu(csr_matrix)
        return True
    except ValueError:
        return False

# 示例使用
data = np.array([4, 1, 1, 3])
row_indices = np.array([0, 0, 1, 1])
col_indices = np.array([0, 1, 0, 1])
csr_mat = csr_matrix((data, (row_indices, col_indices)), shape=(2, 2))

print(is_positive_definite(csr_mat))  # 输出: True