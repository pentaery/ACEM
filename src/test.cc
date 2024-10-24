#include <cmath>
#include <cstdio>
#include <mkl.h>
#include <vector>
#include "mkl_spblas.h"
#include "system.hh"

int main() {
  int size = 512;
  sparse_status_t status;
  std::vector<precision> rhs(size * size);
  formRHS(rhs, size);

  sparse_matrix_t A;
  status = mkl_sparse_s_set_value(A, 1, 1, 0.01);
  return 0;
}