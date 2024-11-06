#include "mkl_pardiso.h"
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

  printf("exportcsr\n");


  _MKL_DSS_HANDLE_t pt = nullptr;
  MKL_INT maxfct = 1, mnum = 1, mtype = 2, phase = 22;
  MKL_INT n = size * size;
  pardiso(pt, &maxfct, &mnum, &mtype, &phase,
          &n, const void *, const int *, const int *, int *,
          const int *, int *, const int *, void *, void *, int *);

  pardisoinit(_MKL_DSS_HANDLE_t, const int *, int *);
  
  clock_t end = clock();
  printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
  return 0;
}