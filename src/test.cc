#include "system.hh"
#include <cstdio>

int main() {
  int size = 512;
  sparse_status_t status;
  std::vector<precision> rhs(size * size);
  formRHS(rhs, size);

  sparse_matrix_t A;
  formA(A, size);



  return 0;
}