#include "system.hh"
#include <cstdio>
#include <iostream>

#include <vector>

void System::formRHS() {
  double gridLength = 1.0 / (size - 1);
  for (int row = 0; row < size; ++row) {
    for (int col = 0; col < size; ++col) {
      vecRHS[row * size + col] = 2 * M_PI * M_PI *
                                 sin(M_PI * (row + 1) * gridLength) *
                                 sin(M_PI * (col + 1) * gridLength);
    }
  }
}

void System::getData() {
  std::vector<MKL_INT> row_indx;
  std::vector<MKL_INT> col_indx;
  std::vector<double> values;
  values.reserve(5 * size * size);
  for (MKL_INT row = 0; row < size; ++row) {
    for (MKL_INT col = 0; col < size; ++col) {
      MKL_INT index = row * size + col;
      if (col + row == 0 || col + row == 2 * size - 2) {
        row_indx.push_back(index);
        col_indx.push_back(index);
        values.push_back(2.0);
      } else if (col + row == size - 1) {
        row_indx.push_back(index);
        col_indx.push_back(index);
        values.push_back(1.0);
      } else {
        row_indx.push_back(index);
        col_indx.push_back(index);
        values.push_back(0.0);
      }
      if (row > 0) {
        row_indx.push_back(index);
        col_indx.push_back((row - 1) * size + col);
        values.push_back(1.0);
      }
      if (row < size - 1) {
        row_indx.push_back(index);
        col_indx.push_back((row + 1) * size + col);
        values.push_back(1.0);
      }
      if (col > 0) {
        row_indx.push_back(index);
        col_indx.push_back(index - 1);
        values.push_back(1.0);
      }
      if (col < size - 1) {
        row_indx.push_back(index);
        col_indx.push_back(index + 1);
        values.push_back(1.0);
      }
    }
  }
  std::cout << "non-zero elements in L: " << values.size() << std::endl;
  // int i;
  // for (i = 0; i < values.size(); ++i) {
  //   std::cout << row_indx[i] << " " << col_indx[i] << " " << values[i]
  //             << std::endl;
  // }
  sparse_matrix_t matB;
  mkl_sparse_d_create_coo(&matB, indexing, nvtxs, nvtxs, values.size(),
                          row_indx.data(), col_indx.data(), values.data());
  mkl_sparse_convert_csr(matB, SPARSE_OPERATION_NON_TRANSPOSE, &matL);
  mkl_sparse_destroy(matB);
}

void System::graphPartition() {
  MKL_INT *rows_start, *rows_end, *col_index;
  mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  rows_start[size * size] = rows_end[size * size - 1];

  idx_t ncon = 1;
  idx_t objval;
  idx_t options[METIS_NOPTIONS];
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
  options[METIS_OPTION_NCUTS] = 1;
  METIS_PartGraphKway(&nvtxs, &ncon, rows_start, col_index, NULL, NULL, NULL,
                      &nparts, NULL, NULL, NULL, &objval, part);
  std::cout << "Finish partitioning with objval " << objval << std::endl;

  FILE *fp;
  if ((fp = fopen("../../partition.txt", "wb")) == NULL) {
    printf("cant open the file");
    exit(0);
  }
  for (int i = 0; i < nvtxs; i++) {
    fprintf(fp, "%d ", part[i]);
  }
  fclose(fp);
}

void System::formA() {
  MKL_INT *rows_start, *rows_end, *col_index;
  MKL_INT rows, cols;
  sparse_index_base_t indexing;
  mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  std::vector<MKL_INT> A_row_index;
  std::vector<MKL_INT> A_col_index;
  std::vector<double> A_values;
  int i = 0, j = 0;
  double diagnal = 0.0;
  for (i = 0; i < nvtxs; ++i) {
    diagnal = 0.0;
    for (j = rows_start[i]; j < rows_end[i]; ++j) {
      if (col_index[j] > i) {
        A_row_index.push_back(i);
        A_col_index.push_back(col_index[j]);
        A_values.push_back(-val[j]);
        diagnal += val[j];
      } else {
        diagnal += val[j];
      }
    }
    A_row_index.push_back(i);
    A_col_index.push_back(i);
    A_values.push_back(diagnal);
  }
  std::cout << "non-zero elements in A: " << A_values.size() << std::endl;
  // for (i = 0; i < A_values.size(); ++i) {
  //   std::cout << A_row_index[i] << " " << A_col_index[i] << " " <<
  //   A_values[i]
  //             << std::endl;
  // }
  sparse_matrix_t matB;
  mkl_sparse_d_create_coo(&matB, indexing, nvtxs, nvtxs, A_values.size(),
                          A_row_index.data(), A_col_index.data(),
                          A_values.data());
  mkl_sparse_convert_csr(matB, SPARSE_OPERATION_NON_TRANSPOSE, &matA);
  mkl_sparse_destroy(matB);
}

void System::solve() {
  MKL_INT *rows_start, *rows_end, *col_index;
  mkl_sparse_d_export_csr(matA, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  rows_start[nvtxs] = rows_end[nvtxs - 1];

  MKL_INT error;

  MKL_INT maxfct = 1, mnum = 1, mtype = 2, phase = 13;
  MKL_INT nrhs = 1, msglv1 = 1;

  MKL_INT idum;
  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &nvtxs, val, rows_start,
          col_index, perm, &nrhs, iparm, &msglv1, vecRHS.data(), vecSOL.data(),
          &error);
  printf("error: %d\n", error);
}

void System::formAUX() {
  char which = 'L';
  MKL_INT pm[128];
  for (int i = 0; i < 128; ++i) {
    pm[i] = 0;
  }
  pm[1] = 6;


  for(int i = 0; i < nvtxs; ++i) {
    
  }

  // mkl_sparse_d_gv(&which, int *pm, sparse_matrix_t A, struct matrix_descr
  // descrA, sparse_matrix_t B, struct matrix_descr descrB, int k0, int *k,
  // double *E, double *X, double *res)
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
  vecSOL.resize(size * size);
  vecRHS.resize(size * size);
  matM.resize(size * size);
  nparts = 10;
  part = new int[nvtxs];
}

System::System(MKL_INT size) {
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
  vecSOL.resize(size * size);
  vecRHS.resize(size * size);
  matM.resize(size * size);
  nparts = 10;
  part = new int[nvtxs];
}

System::System(MKL_INT size, idx_t nparts) {
  this->size = size;
  this->nparts = nparts;
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
  vecSOL.resize(size * size);
  vecRHS.resize(size * size);
  matM.resize(size * size);
  part = new int[nvtxs];
}

//     fprintf(fp, "%d ", part[i]);
//   }
//   fclose(fp);
// }