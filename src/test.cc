#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "system.hh"
#include <bits/types/clock_t.h>
#include <chrono>
#include <cstdio>
#include <vector>

int main() {
  clock_t start = clock();
  int size = 512;
  sparse_status_t status;
  std::vector<float> rhs(size * size);
  std::vector<float> sol(size * size);
  formRHS(rhs, size);

  sparse_matrix_t A;
  formA(A, size);

  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  int rows, cols;
  int *rows_start, *rows_end, *col_index;
  float *val;
  std::vector<int> rowstat(size);

  mkl_sparse_s_export_csr(A, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);

  rows_start[size * size] = rows_end[size * size - 1];
  // printf("lastcol: %d\n", rows_start[size * size - 1]);
  printf("exportcsr\n");

  _MKL_DSS_HANDLE_t pt = nullptr;
  MKL_INT maxfct = 1, mnum = 1, mtype = 2, phase = 11;
  MKL_INT n = size * size;
  MKL_INT nrhs = 1, msglv1 = 1;
  MKL_INT *perm, *iparm;
  MKL_INT error;

  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, &val, rows_start, col_index,
          perm, &nrhs, iparm, &msglv1, &rhs[0], &sol[0], &error);

  clock_t end = clock();
  printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
  return 0;
}
