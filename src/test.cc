#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "system.hh"
#include <bits/types/clock_t.h>
#include <chrono>
#include <cstdio>
#include <vector>

int main() {
  clock_t start = clock();
  int size = 2;
  sparse_status_t status;
  std::vector<float> rhs(8);
  std::vector<float> sol(8);
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

  rows_start[size * size] = rows_end[size * size - 1];
  // printf("lastcol: %d\n", rows_start[size * size - 1]);
  printf("exportcsr\n");
  int i = 0;
  for (i = 0; i < 12; i++) {
    printf("%f ", val[i]);
  }
  printf("\n");
  for (i = 0; i < 12; i++) {
    // col_index[i] = col_index[i] + 1;
    printf("%d ", col_index[i]);
  }
  printf("\n");
  for (i = 0; i < 5; i++) {
    // rows_start[i] = rows_start[i] + 1;
    printf("%d ", rows_start[i]);
  }

  void *pt[64];
  for (i = 0; i < 64; i++) {
    pt[i] = 0;
  }
  MKL_INT maxfct = 1, mnum = 1, mtype = -2, phase = 13;
  MKL_INT n = 4;
  MKL_INT nrhs = 1, msglv1 = 1;
  MKL_INT perm[64], iparm[64];
  MKL_INT idum;
  for (i = 0; i < 64; i++) {
    iparm[i] = 0;
  }
  // iparm[0] = 1;  /* No solver default */
  // iparm[1] = 2;  /* Fill-in reordering from METIS */
  // iparm[3] = 0;  /* No iterative-direct algorithm */
  // iparm[4] = 0;  /* No user fill-in reducing permutation */
  // iparm[5] = 0;  /* Write solution into x */
  // iparm[6] = 0;  /* Not in use */
  // iparm[7] = 2;  /* Max numbers of iterative refinement steps */
  // iparm[8] = 0;  /* Not in use */
  // iparm[9] = 13; /* Perturb the pivot elements with 1E-13 */
  // iparm[10] = 1; /* Use nonsymmetric permutation and scaling MPS */
  // iparm[11] = 0; /* Not in use */
  // iparm[12] =
  //     0; /* Maximum weighted matching algorithm is switched-off (default for
  //           symmetric). Try iparm[12] = 1 in case of inappropriate accuracy
  //           */
  // iparm[13] = 0;  /* Output: Number of perturbed pivots */
  // iparm[14] = 0;  /* Not in use */
  // iparm[15] = 0;  /* Not in use */
  // iparm[16] = 0;  /* Not in use */
  // iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
  // iparm[18] = -1; /* Output: Mflops for LU factorization */
  // iparm[19] = 0;  /* Output: Numbers of CG Iterations */
  MKL_INT ia[5] = {0, 3, 6, 9, 12};
  MKL_INT ja[12] = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
  for (i = 0; i < 5; i++) {
    ia[i] = ia[i] + 1;
  }
  for (i = 0; i < 12; i++) {
    ja[i] = ja[i] + 1;
  }
  float a[12] = {36, -9, -9, -9, 36, -9, -9, 36, -9, -9, -9, 36};
  MKL_INT error;

  double ddum;
  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
          iparm, &msglv1, &rhs[0], &sol[0], &error);
  for (i = 0; i < 4; i++) {
    printf("%f ", rhs[i]);
  }
  printf("\n");
  for (i = 0; i < 4; i++) {
    printf("%f ", sol[i]);
  }
  printf("\n");
  clock_t end = clock();
  printf("Time: %f\n", (double)(end - start) / CLOCKS_PER_SEC);
  return 0;
}
