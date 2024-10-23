#include "system.hpp"
#include <cstdio>
#include <Eigen/IterativeLinearSolvers>

int main(int argc, char **argv) {
  int size = 512;
  SpMat A(pow(size, 2), pow(size, 2));
  SpVec f(pow(size, 2));
  SpVec x(pow(size, 2));
  formSystem(A, f, size);
  printf("A has %ld non-zero entries\n", A.nonZeros());
  printf("f has %ld non-zero entries\n", f.nonZeros());
  Eigen::ConjugateGradient<SpMat, Eigen::Upper> solver;
  solver.compute(A);
  x = solver.solve(f);
  printf("Solved\n");
  printf("Error: %f\n", (A * x - f).norm());
  return 0;
}
