#include "system.hh"
#include <cmath>
#include <cstdio>

void formRHS(std::vector<float> &vec, MKL_INT size) {
  float gridLength = 1.0 / (size + 1);
  MKL_INT row = 0, col = 0;
  for (row = 0; row < size; ++row) {
    for (col = 0; col < size; ++col) {
      vec[row * size + col] = 2 * M_PI * M_PI * sin(M_PI * (row + 1) * gridLength) * sin(M_PI * (col + 1) * gridLength);
    }
  }
}

void formA(sparse_matrix_t &A, MKL_INT size) {
  std::vector<MKL_INT> row_indx;
  std::vector<MKL_INT> col_indx;
  std::vector<float> values;
  float gridLength = 1.0 / (size + 1);
  MKL_INT row = 0, col = 0;
  for (row = 0; row < size; ++row) {
    for (col = 0; col < size; ++col) {
      row_indx.push_back(row * size + col);
      col_indx.push_back(row * size + col);
      values.push_back(4.0 / gridLength / gridLength);
      if(row > 0) {
        row_indx.push_back(row * size + col);
        col_indx.push_back((row - 1) * size + col);
        values.push_back(-1.0 / gridLength / gridLength);
      }
      if(row < size - 1) {
        row_indx.push_back(row * size + col);
        col_indx.push_back((row + 1) * size + col);
        values.push_back(-1.0 / gridLength / gridLength);
      }
      if(col > 0) {
        row_indx.push_back(row * size + col);
        col_indx.push_back(row * size + col - 1);
        values.push_back(-1.0 / gridLength / gridLength);
      }
      if(col < size - 1) {
        row_indx.push_back(row * size + col);
        col_indx.push_back(row * size + col + 1);
        values.push_back(-1.0 / gridLength / gridLength);
      }
    }
  }
  printf("size: %ld\n", values.size());
  sparse_matrix_t B;
  mkl_sparse_s_create_coo(&B, SPARSE_INDEX_BASE_ZERO, size * size, size * size,
                          values.size(), &row_indx[0], &col_indx[0],
                          &values[0]);

  mkl_sparse_convert_csr(B, SPARSE_OPERATION_NON_TRANSPOSE, &A);
}