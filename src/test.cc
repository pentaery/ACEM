#include <iostream>
#include <mkl.h>

int main() {
  // 定义稀疏矩阵的大小
  int rows = 3;
  int cols = 3;

  // CSR 格式的数组
  MKL_INT row_index[] = {0, 2, 3, 3}; // 行指针
  MKL_INT col_index[] = {0, 2, 1};    // 列索引
  double values[] = {1.0, 2.0, 3.0};  // 非零值

  // 创建稀疏矩阵描述符
  sparse_matrix_t A;
  mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ONE, rows, cols, row_index,
                          row_index + 1, col_index, values);

  // 创建一个向量用于乘法
  double x[3] = {1.0, 2.0, 3.0}; // 输入向量
  double y[3];                   // 输出向量

  // 执行稀疏矩阵与向量的乘法
//   mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, 2.0, x, 0.0, y);

//   // 输出结果
//   std::cout << "Result of sparse matrix-vector multiplication:" << std::endl;
//   for (int i = 0; i < rows; ++i) {
//     std::cout << y[i] << " ";
//   }
//   std::cout << std::endl;

  // 释放稀疏矩阵
  mkl_sparse_destroy(A);

  return 0;
}