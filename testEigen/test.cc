#define EIGEN_USE_MKL_ALL
#include "system.hh"
#include <cstdio>
#include <chrono>
#include <Eigen/PardisoSupport>

int main(int argc, char **argv) {
  auto start = std::chrono::high_resolution_clock::now();
  int size = 512;
  SpMat A(pow(size, 2), pow(size, 2));
  SpVec f(pow(size, 2));
  SpVec x(pow(size, 2));
  formSystem(A, f, size);
  printf("A has %ld non-zero entries\n", A.nonZeros());
  printf("f has %ld non-zero entries\n", f.nonZeros());
  // Eigen::SimplicialLLT<SpMat> solver;
  Eigen::PardisoLLT<SpMat> solver;
  solver.compute(A);
  x = solver.solve(f);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  printf("Time elapsed: %f ms\n", duration.count());
  return 0;
}
