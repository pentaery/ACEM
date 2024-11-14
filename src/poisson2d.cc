#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "system.hh"
#include <bits/types/clock_t.h>
#include <chrono>
#include <cstdio>
#include <metis.h>
#include <vector>

int main() {
  auto start = std::chrono::high_resolution_clock::now();

  System sys(1024);
  sys.formRHS();
  sys.formA();
  // sys.solve();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  printf("Time elapsed: %f ms\n", duration.count());

  return 0;
}