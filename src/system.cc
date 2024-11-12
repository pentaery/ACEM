#include "system.hh"
#include "mkl_spblas.h"
#include <cmath>
#include <cstdio>
#include <iostream>

void System::formRHS() {
  rhs.resize(size * size);
  double gridLength = 1.0 / (size + 1);
  for (MKL_INT row = 0; row < size; ++row) {
    for (MKL_INT col = 0; col < size; ++col) {
      rhs[row * size + col] = 2 * M_PI * M_PI *
                              sin(M_PI * (row + 1) * gridLength) *
                              sin(M_PI * (col + 1) * gridLength);
    }
  }
}

void System::formA() {
  double gridLength = 1.0 / (size + 1);

  for (MKL_INT row = 0; row < size; ++row) {
    for (MKL_INT col = 0; col < size; ++col) {
      MKL_INT index = row * size + col;
      row_indx.push_back(index);
      col_indx.push_back(index);
      values.push_back(4.0 / (gridLength * gridLength));
      if (row < size - 1) {
        row_indx.push_back(index);
        col_indx.push_back((row + 1) * size + col);
        values.push_back(-1.0 / (gridLength * gridLength));
      }
      if (col < size - 1) {
        row_indx.push_back(index);
        col_indx.push_back(index + 1);
        values.push_back(-1.0 / (gridLength * gridLength));
      }
    }
  }
  printf("size: %ld\n", values.size());
  sparse_matrix_t B;
  mkl_sparse_d_create_coo(&B, indexing, size * size, size * size, values.size(),
                          row_indx.data(), col_indx.data(), values.data());

  mkl_sparse_convert_csr(B, SPARSE_OPERATION_NON_TRANSPOSE, &A);

  mkl_sparse_d_export_csr(A, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  rows_start[size * size] = rows_end[size * size - 1];

  printf("exportcsr\n");
}

System::System() {
  int i;
  for (i = 0; i < 64; i++) {
    pt[i] = 0;
  }
  for (i = 0; i < 64; i++) {
    iparm[i] = 0;
  }
  iparm[34] = 1;
  iparm[0] = 1;
  sol.resize(size * size);
}

System::System(int size) {
  this->size = size;
  int i;
  for (i = 0; i < 64; i++) {
    pt[i] = 0;
  }
  for (i = 0; i < 64; i++) {
    iparm[i] = 0;
  }
  iparm[34] = 1;
  iparm[0] = 1;
}

void System::solve() {

  MKL_INT error;

  MKL_INT maxfct = 1, mnum = 1, mtype = 2, phase = 13;
  MKL_INT n = size * size;
  MKL_INT nrhs = 1, msglv1 = 1;

  MKL_INT idum;
  printf("iparm[0]: %d\n", iparm[0]);
  printf("iparm[34]: %d\n", iparm[34]);
  std::cout<<sol.data()<<std::endl;
  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, val, rows_start, col_index,
          perm, &nrhs, iparm, &msglv1, rhs.data(), sol.data(), &error);
  printf("error: %d\n", error);
}

void formRHS(std::vector<double> &vec, MKL_INT size) {
  double gridLength = 1.0 / (size + 1);
  MKL_INT row = 0, col = 0;
  for (row = 0; row < size; ++row) {
    for (col = 0; col < size; ++col) {
      vec[row * size + col] = 2 * M_PI * M_PI *
                              sin(M_PI * (row + 1) * gridLength) *
                              sin(M_PI * (col + 1) * gridLength);
    }
  }
}

void formA(sparse_matrix_t &A, MKL_INT size) {
  std::vector<MKL_INT> row_indx;
  std::vector<MKL_INT> col_indx;
  std::vector<double> values;
  double gridLength = 1.0 / (size + 1);
  MKL_INT row = 0, col = 0;
  for (row = 0; row < size; ++row) {
    for (col = 0; col < size; ++col) {
      row_indx.push_back(row * size + col);
      col_indx.push_back(row * size + col);
      values.push_back(4.0 / gridLength / gridLength);
      if (row < size - 1) {
        row_indx.push_back(row * size + col);
        col_indx.push_back((row + 1) * size + col);
        values.push_back(-1.0 / gridLength / gridLength);
      }
      if (col < size - 1) {
        row_indx.push_back(row * size + col);
        col_indx.push_back(row * size + col + 1);
        values.push_back(-1.0 / gridLength / gridLength);
      }
    }
  }
  printf("size: %ld\n", values.size());
  sparse_matrix_t B;
  mkl_sparse_d_create_coo(&B, SPARSE_INDEX_BASE_ZERO, size * size, size * size,
                          values.size(), &row_indx[0], &col_indx[0],
                          &values[0]);

  mkl_sparse_convert_csr(B, SPARSE_OPERATION_NON_TRANSPOSE, &A);
}