#include <cmath>
#include <cstdio>
#include <mkl.h>
#include <vector>
#include "system.hh"

int main() {
  int size = 512;
  std::vector<precision> rhs(size * size);
  formRHS(rhs, size);

  sparse_matrix_t A;
  
  return 0;
}