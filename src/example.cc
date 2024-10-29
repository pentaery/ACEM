#include "mkl_lapacke.h"
#include <stdio.h>
#include <stdlib.h>

extern void print_matrix(const char *desc, MKL_INT m, MKL_INT n, float *a,
                         MKL_INT lda);
extern void print_int_vector(const char *desc, MKL_INT n, MKL_INT *a);

#define N 5
#define NRHS 3
#define LDA N
#define LDB NRHS

int main() {

  MKL_INT n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info;

  MKL_INT ipiv[N];
  float a[LDA * N] = {6.80f,  -6.05f, -0.45f, 8.32f, -9.67f, -2.11f, -3.30f,
                      2.58f,  2.71f,  -5.14f, 5.66f, 5.36f,  -2.70f, 4.35f,
                      -7.26f, 5.97f,  -4.44f, 0.27f, -7.17f, 6.08f,  8.23f,
                      1.08f,  9.04f,  2.14f,  -6.87f};
  float b[LDB * N] = {4.02f,  -1.56f, 9.81f,  6.19f,  4.00f,
                      -4.09f, -8.22f, -8.67f, -4.57f, -7.57f,
                      1.75f,  -8.61f, -3.03f, 2.86f,  8.99f};

  printf("LAPACKE_sgesv (row-major, high-level) Example Program Results\n");

  info = LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv, b, ldb);

  if (info > 0) {
    printf("The diagonal element of the triangular factor of A,\n");
    printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
    printf("the solution could not be computed.\n");
    exit(1);
  }

  print_matrix("Solution", n, nrhs, b, ldb);

  print_matrix("Details of LU factorization", n, n, a, lda);

  print_int_vector("Pivot indices", n, ipiv);
  exit(0);
}

void print_matrix(const char *desc, MKL_INT m, MKL_INT n, float *a,
                  MKL_INT lda) {
  MKL_INT i, j;
  printf("\n %s\n", desc);
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++)
      printf(" %6.2f", a[i * lda + j]);
    printf("\n");
  }
}

void print_int_vector(const char *desc, MKL_INT n, MKL_INT *a) {
  MKL_INT j;
  printf("\n %s\n", desc);
  for (j = 0; j < n; j++)
    printf(" %6i", a[j]);
  printf("\n");
}