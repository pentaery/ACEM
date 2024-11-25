#include "system.hh"
#include "mkl_spblas.h"
#include <cstdio>
#include <iostream>

#include <ostream>
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
    for (j = rows_start[i]; j < rows_end[i]; ++j) {
      if (col_index[j] < i) {
        A_row_index.push_back(i);
        A_col_index.push_back(i);
        A_values.push_back(val[j]);
      } else if (col_index[j] > i) {
        A_row_index.push_back(i);
        A_col_index.push_back(i);
        A_values.push_back(val[j]);
        A_row_index.push_back(i);
        A_col_index.push_back(col_index[j]);
        A_values.push_back(-val[j]);
      } else {
        A_row_index.push_back(i);
        A_col_index.push_back(i);
        A_values.push_back(val[j]);
      }
    }
  }
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

  MKL_INT error;

  MKL_INT maxfct = 1, mnum = 1, mtype = 2, phase = 13;
  MKL_INT nrhs = 1, msglv1 = 1;

  MKL_INT idum;
  pardiso(pt, &maxfct, &mnum, &mtype, &phase, &nvtxs, val, rows_start,
          col_index, perm, &nrhs, iparm, &msglv1, vecRHS.data(), vecSOL.data(),
          &error);
  printf("error: %d\n", error);
}

void System::findNeighbours() {
  int i = 0, j = 0;
  for (i = 0; i < nvtxs; ++i) {
    vertices[part[i]].insert(i);
  }
  MKL_INT *rows_start, *rows_end, *col_index;
  mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  for (i = 0; i < nvtxs; ++i) {
    for (j = rows_start[i]; j < rows_end[i]; ++j) {
      if (part[i] != part[col_index[j]]) {
        neighbours[part[i]].insert(part[col_index[j]]);
      }
    }
  }
  for (i = 0; i < overlap; ++i) {
    for (j = 0; j < nparts; ++j) {
      if (i == 0) {
        overlapping[j].insert(neighbours[j].begin(), neighbours[j].end());
      } else {
        for (const auto &element : overlapping[j]) {
          overlapping[j].insert(neighbours[element].begin(),
                                neighbours[element].end());
        }
      }
    }
  }

  globalTolocal.resize(nvtxs);

  count.resize(nparts);
  for (i = 0; i < nparts; ++i) {
    count[i] = 0;
  }
  for (i = 0; i < nvtxs; ++i) {
    globalTolocal[i] = count[part[i]];
    count[part[i]]++;
  }

  // for (const auto &element : overlapping[0]) {
  //   std::cout << element << " ";
  // }
  // std::cout << std::endl;
}

void System::formAUX() {
  MKL_INT *rows_start, *rows_end, *col_index;
  MKL_INT rows, cols;
  sparse_index_base_t indexing;
  mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  int i = 0, j = 0;

  char which = 'S';
  MKL_INT pm[128];

  mkl_sparse_ee_init(pm);
  std::cout << "nparts: " << nparts << std::endl;

  std::vector<std::vector<MKL_INT>> Ai_col_index;
  std::vector<std::vector<MKL_INT>> Ai_row_index;
  std::vector<std::vector<double>> Ai_values;
  std::vector<std::vector<MKL_INT>> Si_col_index;
  std::vector<std::vector<MKL_INT>> Si_row_index;
  std::vector<std::vector<double>> Si_values;
  Ai_col_index.resize(nparts);
  Ai_row_index.resize(nparts);
  Ai_values.resize(nparts);
  Si_col_index.resize(nparts);
  Si_row_index.resize(nparts);
  Si_values.resize(nparts);
  for (i = 0; i < nparts; ++i) {
    Ai_col_index[i].reserve(nvtxs);
    Ai_row_index[i].reserve(nvtxs);
    Ai_values[i].reserve(nvtxs);
    Si_col_index[i].reserve(nvtxs);
    Si_row_index[i].reserve(nvtxs);
    Si_values[i].reserve(nvtxs);
  }

  for (i = 0; i < nvtxs; ++i) {
    for (j = rows_start[i]; j < rows_end[i]; ++j) {
      if (part[i] == part[col_index[j]]) {
        if (col_index[j] != i) {
          Ai_row_index[part[i]].push_back(globalTolocal[i]);
          Ai_col_index[part[i]].push_back(globalTolocal[i]);
          Ai_values[part[i]].push_back(val[j]);
          Ai_row_index[part[i]].push_back(globalTolocal[i]);
          Ai_col_index[part[i]].push_back(globalTolocal[col_index[j]]);
          Ai_values[part[i]].push_back(-val[j]);

          Si_col_index[part[i]].push_back(globalTolocal[i]);
          Si_row_index[part[i]].push_back(globalTolocal[i]);
          Si_values[part[i]].push_back(val[j] / cStar / cStar / 2);
        } else {
          Ai_row_index[part[i]].push_back(globalTolocal[i]);
          Ai_col_index[part[i]].push_back(globalTolocal[i]);
          Ai_values[part[i]].push_back(val[j]);

          Si_col_index[part[i]].push_back(globalTolocal[i]);
          Si_row_index[part[i]].push_back(globalTolocal[i]);
          Si_values[part[i]].push_back(val[j] / cStar / cStar);
        }
      }
    }
  }

  std::vector<sparse_matrix_t> AiCOO;
  std::vector<sparse_matrix_t> SiCOO;
  std::vector<sparse_matrix_t> Ai;
  std::vector<sparse_matrix_t> Si;

  AiCOO.resize(nparts);
  SiCOO.resize(nparts);
  Ai.resize(nparts);
  Si.resize(nparts);

  int k0 = 3;
  int k;

  eigenvalue.resize(nparts);
  for (i = 0; i < nparts; ++i) {
    eigenvalue[i].resize(k0);
  }

  eigenvector.resize(nparts);
  for (i = 0; i < nparts; ++i) {
    eigenvector[i].resize(k0 * count[i]);
  }

  double E[3][4];
  double X[3][12];

  double res[nparts];

  matrix_descr descr;

  descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
  descr.diag = SPARSE_DIAG_NON_UNIT;
  descr.mode = SPARSE_FILL_MODE_UPPER;

  // #pragma omp parallel for
  for (i = 0; i < nparts; ++i) {
    mkl_sparse_d_create_coo(&AiCOO[i], indexing, count[i], count[i],
                            Ai_values[i].size(), Ai_row_index[i].data(),
                            Ai_col_index[i].data(), Ai_values[i].data());
    mkl_sparse_d_create_coo(&SiCOO[i], indexing, count[i], count[i],
                            Si_values[i].size(), Si_row_index[i].data(),
                            Si_col_index[i].data(), Si_values[i].data());

    mkl_sparse_convert_csr(AiCOO[i], SPARSE_OPERATION_NON_TRANSPOSE, &Ai[i]);
    mkl_sparse_convert_csr(SiCOO[i], SPARSE_OPERATION_NON_TRANSPOSE, &Si[i]);

    mkl_sparse_destroy(AiCOO[i]);
    mkl_sparse_destroy(SiCOO[i]);

    sparse_status_t error =
        mkl_sparse_d_gv(&which, pm, Ai[i], descr, Si[i], descr, k0, &k,
                        eigenvalue[i].data(), eigenvector[i].data(), &res[i]);
    if (error != 0) {
      std::cout << "===========error in solving eigenvalue problem==========="
                << error << "===============" << std::endl;
    }
    if (k < k0) {
      std::cout << "===========Not find enough eigenvalues==========="
                << std::endl;
    }

    mkl_sparse_destroy(Ai[i]);
    mkl_sparse_destroy(Si[i]);
  }
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
  vertices.resize(nparts);
  neighbours.resize(nparts);
  overlapping.resize(nparts);
  overlap = 2;
  cStar = 1.0;
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
  vertices.resize(nparts);
  neighbours.resize(nparts);
  overlapping.resize(nparts);
  overlap = 2;
  cStar = 1.0;
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
  vertices.resize(nparts);
  neighbours.resize(nparts);
  overlapping.resize(nparts);
  overlap = 2;
  cStar = 1.0;
}

System::System(MKL_INT size, idx_t nparts, int overlap) {
  this->overlap = overlap;
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
  vertices.resize(nparts);
  neighbours.resize(nparts);
  overlapping.resize(nparts);
  cStar = 1.0;
}
//     fprintf(fp, "%d ", part[i]);
//   }
//   fclose(fp);
// }