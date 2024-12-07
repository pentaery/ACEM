#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "system.hh"

#include <cstdio>
#include <metis.h>
#include <vector>

int main() {
  auto start = std::chrono::high_resolution_clock::now();

  System sys(20, 20, 2);
  sys.getData();
  sys.graphPartition();
  sys.formRHS();
  // sys.formA();
  // sys.solve();
  sys.findNeighbours();
  sys.formAUX();
  sys.formCEM();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  printf("Time elapsed: %f ms\n", duration.count());

  return 0;
}