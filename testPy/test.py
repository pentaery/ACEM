import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# 开始计时
import time
start_time = time.time()

size = 512
lx = 1.0
ly = 1.0
hx = lx / (size + 1)
hy = ly / (size + 1)
f = np.zeros(size**2)
A = lil_matrix((size**2, size**2))  # 使用 LIL 格式构建稀疏矩阵

for row in range(size):
    for col in range(size):
        A[row * size + col, row * size + col] = 4.0 / (hx * hy)

        if row > 0:
            A[row * size + col, (row - 1) * size + col] = -1.0 / (hx * hy)

        if row < size - 1:
            A[row * size + col, (row + 1) * size + col] = -1.0 / (hx * hy)

        if col > 0:
            A[row * size + col, row * size + col - 1] = -1.0 / (hx * hy)

        if col < size - 1:
            A[row * size + col, row * size + col + 1] = -1.0 / (hx * hy)

        f[row * size + col] = 2 * np.pi**2 * np.sin(np.pi * (col + 1) * hx) * np.sin(np.pi * (row + 1) * hy)

print('Finish loop')

# 将稀疏矩阵转换为 CSR 格式以进行高效计算
A = A.tocsr()

print('Finish A')

# 求解 Ax = f
x = spsolve(A, f)

# 输出运行时间
end_time = time.time()
print(f'Time taken: {end_time - start_time:.2f} seconds')