#include "mkl_pardiso.h"
#include "system.hh"
#include <bits/types/clock_t.h>
#include <chrono>
#include <cstdio>

int main() {
  clock_t start = clock();
  int size = 512;
  sparse_status_t status;
  std::vector<float> rhs(size * size);
  formRHS(rhs, size);

  sparse_matrix_t A;
  formA(A, size);
  _MKL_DSS_HANDLE_t pt = 0;
  pardiso(pt, const int *maxfct, const int *mnum, const int *mtype,
          const int *phase, const int *n, const void *a, const int *ia,
          const int *ja, int *perm, const int *nrhs, int *iparm,
          const int *msglvl, void *b, void *x, int *error)

      clock_t end = clock();
  printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
  return 0;
}