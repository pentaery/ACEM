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

  std::vector<MKL_INT> A_row_index = {0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2};
  std::vector<MKL_INT> A_col_index = {0, 0, 1, 1, 0, 1, 1, 2, 2, 1, 2};
  std::vector<double> A_values = {2, 1, -1, 1, -1, 0, 1, -1, 1, -1, 1};

  std::vector<MKL_INT> A2_row_index = {0, 0, 1, 1, 1, 2, 2};
  std::vector<MKL_INT> A2_col_index = {0, 1, 0, 1, 2, 1, 2};
  std::vector<double> A2_values = {3, -1, -1, 2, -1, -1, 2};

  std::vector<MKL_INT> S_row_index = {0, 0, 1, 1, 1, 2, 2};
  std::vector<MKL_INT> S_col_index = {0, 0, 1, 1, 1, 2, 2};
  std::vector<double> S_values = {2, 0.5, 0.5, 0, 0.5, 0.5, 1};

  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  sparse_matrix_t Ai, Si;
  sparse_matrix_t AiCOO = NULL, SiCOO = NULL;
  mkl_sparse_d_create_coo(&AiCOO, indexing, 3, 3, 11, A_row_index.data(),
                          A_col_index.data(), A_values.data());
  mkl_sparse_d_create_coo(&SiCOO, indexing, 3, 3, 7, S_row_index.data(),
                          S_col_index.data(), S_values.data());

  mkl_sparse_convert_csr(AiCOO, SPARSE_OPERATION_NON_TRANSPOSE, &Ai);
  mkl_sparse_convert_csr(SiCOO, SPARSE_OPERATION_NON_TRANSPOSE, &Si);

  int rows, cols;
  int *rows_start, *rows_end, *col_indx;
  double *values;

  mkl_sparse_d_export_csr(Ai, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_indx, &values);
  int i;
  for (i = 0; i < 7; ++i) {
    std::cout << rows_start[i] << " ";
  }
  std::cout << std::endl;
  for (i = 0; i < 7; ++i) {
    std::cout << col_indx[i] << " ";
  }
  std::cout << std::endl;

  for (i = 0; i < 7; ++i) {
    std::cout << values[i] << " ";
  }
  std::cout << std::endl;

  char which = 'S';
  MKL_INT pm[128];
  mkl_sparse_ee_init(pm);
  // pm[1] = 7;
  // pm[2] = 2;
  // pm[3] = 10;
  // pm[4] = 10000;
  // pm[6] = 1;
  // pm[8] = 1;
  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
  descr.mode = SPARSE_FILL_MODE_UPPER;
  descr.diag = SPARSE_DIAG_NON_UNIT;
  int k0 = 3;
  int k;
  double E[3];
  double X[9];
  double res;

  sparse_status_t error =
      mkl_sparse_d_gv(&which, pm, Ai, descr, Si, descr, k0, &k, E, X, &res);

  std::cout << "error: " << error << std::endl;
  std::cout << "number of found eigenvalues: " << k << std::endl;
  std::cout << "smallest eigenvalue: " << E[0] << " " << E[1] << " " << E[2]
            << std::endl;

  return 0;
}
