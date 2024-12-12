#include "mkl.h"
#include "mkl_solvers_ee.h"
#include "mkl_spblas.h"
#include "mkl_types.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main() {

  int i, j;

  std::vector<MKL_INT> A_row_index = {0, 0, 1, 1, 2};
  std::vector<MKL_INT> A_col_index = {0, 1, 1, 2, 2};
  std::vector<double> A_values = {2, -1, 2, -1, 2};
  std::vector<double> rhs = {1, 0, 0};
  std::vector<double> sol = {0, 0, 0};

  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  sparse_matrix_t Ai;
  sparse_matrix_t AiCOO;
  mkl_sparse_d_create_coo(&AiCOO, indexing, 3, 3, 5, A_row_index.data(),
                          A_col_index.data(), A_values.data());
  mkl_sparse_convert_csr(AiCOO, SPARSE_OPERATION_NON_TRANSPOSE, &Ai);

  int rows, cols;
  int *rows_start, *rows_end, *col_index;
  double *val;

  mkl_sparse_d_export_csr(Ai, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);

  MKL_INT error;

  MKL_INT maxfct = 1, mnum = 1, mtype = 2, phase = 13;
  MKL_INT msglv1 = 1;

  MKL_INT idum;
  MKL_INT perm[64], iparm[64];
  void *pt[64];
  for (j = 0; j < 64; j++) {
    pt[j] = 0;
  }
  for (j = 0; j < 64; j++) {
    iparm[j] = 0;
  }
  iparm[34] = 1;
  iparm[0] = 1;

  int n = 3;
  int k = 1;

  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, val, rows_start, col_index,
          perm, &k, iparm, &msglv1, rhs.data(), sol.data(), &error);


  
  mkl_sparse_destroy(Ai);
  mkl_sparse_destroy(AiCOO);

  return 0;
}
