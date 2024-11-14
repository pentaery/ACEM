#include "system.hh"
#include "mkl_spblas.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <metis.h>

void System::formRHS() {
  double gridLength = 1.0 / (size + 1);
  for (MKL_INT row = 0; row < size; ++row) {
    for (MKL_INT col = 0; col < size; ++col) {
      rhs[row * size + col] = 2 * M_PI * M_PI *
                              sin(M_PI * (row + 1) * gridLength) *
                              sin(M_PI * (col + 1) * gridLength);
    }
  }
}

void System::formA() {
  double gridLength = 1.0 / (size + 1);

  for (MKL_INT row = 0; row < size; ++row) {
    for (MKL_INT col = 0; col < size; ++col) {
      MKL_INT index = row * size + col;
      // row_indx.push_back(index);
      // col_indx.push_back(index);
      // values.push_back(4.0 / (gridLength * gridLength));
      if (row > 0) {
        row_indx.push_back(index);
        col_indx.push_back((row - 1) * size + col);
        values.push_back(-1.0 / (gridLength * gridLength));
      }
      if (row < size - 1) {
        row_indx.push_back(index);
        col_indx.push_back((row + 1) * size + col);
        values.push_back(-1.0 / (gridLength * gridLength));
      }
      if (col > 0) {
        row_indx.push_back(index);
        col_indx.push_back(index - 1);
        values.push_back(-1.0 / (gridLength * gridLength));
      }
      if (col < size - 1) {
        row_indx.push_back(index);
        col_indx.push_back(index + 1);
        values.push_back(-1.0 / (gridLength * gridLength));
      }
    }
  }
  printf("size: %ld\n", values.size());
  sparse_matrix_t B;
  mkl_sparse_d_create_coo(&B, indexing, size * size, size * size, values.size(),
                          row_indx.data(), col_indx.data(), values.data());

  mkl_sparse_convert_csr(B, SPARSE_OPERATION_NON_TRANSPOSE, &A);

  mkl_sparse_d_export_csr(A, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  rows_start[size * size] = rows_end[size * size - 1];

  int i;
  // for (i = 0; i < size * size; i++) {
  //   printf("%d ", rows_start[i]);
  // }
  // printf("\n");
  // for (i = 0; i < size * size; i++) {
  //   printf("%d ", col_index[i]);
  // }
  // printf("\n");

  printf("exportcsr\n");

  idx_t ncon = 1;
  idx_t nparts = 4096;
  idx_t objval;
  idx_t part[nvtxs];
  METIS_PartGraphKway(&nvtxs, &ncon, rows_start, col_index, NULL, NULL, NULL,
                      &nparts, NULL, NULL, NULL, &objval, part);
  printf("Finish partitioning with objval %d\n", objval);

  FILE *fp;
  if ((fp = fopen("../../partition.txt", "wb")) == NULL) {
    printf("cant open the file");
    exit(0);
  }

  for (i = 0; i < nvtxs; i++) {
    fprintf(fp, "%d ", part[i]);
  }
  fclose(fp);
}

System::System() {
  size = 512;
  indexing = SPARSE_INDEX_BASE_ZERO;
  int i;
  for (i = 0; i < 64; i++) {
    pt[i] = 0;
  }
  for (i = 0; i < 64; i++) {
    iparm[i] = 0;
  }
  iparm[34] = 1;
  iparm[0] = 1;
  nvtxs = size * size;
  sol.resize(size * size);
  rhs.resize(size * size);
}

System::System(int size) : System() {
  this->size = size;
  indexing = SPARSE_INDEX_BASE_ZERO;
  int i;
  for (i = 0; i < 64; i++) {
    pt[i] = 0;
  }
  for (i = 0; i < 64; i++) {
    iparm[i] = 0;
  }
  iparm[34] = 1;
  iparm[0] = 1;
  nvtxs = size * size;
  sol.resize(size * size);
  rhs.resize(size * size);
}

void System::solve() {

  MKL_INT error;

  MKL_INT maxfct = 1, mnum = 1, mtype = 2, phase = 13;
  MKL_INT nrhs = 1, msglv1 = 1;

  MKL_INT idum;
  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &nvtxs, val, rows_start,
          col_index, perm, &nrhs, iparm, &msglv1, rhs.data(), sol.data(),
          &error);
  printf("error: %d\n", error);
}

void System::graphPartition() {}
