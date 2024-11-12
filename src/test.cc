#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "system.hh"
#include <bits/types/clock_t.h>
#include <chrono>
#include <cstdio>
#include <vector>

int main() {
  auto start = std::chrono::high_resolution_clock::now();
  int size = 512;
  sparse_status_t status;
  std::vector<double> rhs(size * size);
  std::vector<double> sol(size * size);
  formRHS(rhs, size);

  sparse_matrix_t A;
  formA(A, size);

  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  int rows, cols;
  int *rows_start, *rows_end, *col_index;
  double *val;

  mkl_sparse_d_export_csr(A, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);

  rows_start[size * size] = rows_end[size * size - 1];
  printf("exportcsr\n");

  int i;
  void *pt[64];
  for (i = 0; i < 64; i++) {
    pt[i] = 0;
  }
  MKL_INT maxfct = 1, mnum = 1, mtype = 2, phase = 13;
  MKL_INT n = size * size;
  MKL_INT nrhs = 1, msglv1 = 1;
  MKL_INT perm[64], iparm[64];
  MKL_INT idum;
  for (i = 0; i < 64; i++) {
    iparm[i] = 0;
  }
  iparm[34] = 1;
  iparm[0] = 1; 
  MKL_INT error;

  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, val, rows_start, col_index,
          &idum, &nrhs, iparm, &msglv1, rhs.data(), sol.data(), &error);
  for (i = 0; i < 9; i++) {
    printf("%f ", rhs[i]);
  }
  printf("\n");
  for (i = 0; i < 9; i++) {
    printf("%f ", sol[i]);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  printf("Time elapsed: %f ms\n", duration.count());
  return 0;
}
