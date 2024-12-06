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
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "======Phase I: Graph Partitioning======" << std::endl;
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
  std::cout << "Objective for the partition is " << objval << std::endl;

  FILE *fp;
  if ((fp = fopen("../../partition.txt", "wb")) == NULL) {
    printf("cant open the file");
    exit(0);
  }
  for (int i = 0; i < nvtxs; i++) {
    fprintf(fp, "%d ", part[i]);
  }
  fclose(fp);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Finished with " << duration.count()
            << " ms======" << std::endl;
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
  auto start = std::chrono::high_resolution_clock::now();
  std::cout
      << "======Phase II: Construct the neighours for the CEM method======"
      << std::endl;
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

  if (overlap > 0) {
    for (j = 0; j < nparts; ++j) {
      overlapping[j].insert(neighbours[j].begin(), neighbours[j].end());
      overlapping[j].insert(j);
    }
  }

  for (i = 1; i < overlap; ++i) {
    for (j = 0; j < nparts; ++j) {
      std::set<MKL_INT> tempset;
      for (const auto &element : overlapping[j]) {
        tempset.insert(neighbours[element].begin(), neighbours[element].end());
      }
      overlapping[j].insert(tempset.begin(), tempset.end());
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

  verticesCEM.resize(nparts);
  globalTolocalCEM.resize(nparts);
  for (i = 0; i < nparts; ++i) {
    j = 0;
    for (const auto &element : overlapping[i]) {
      verticesCEM[i].insert(vertices[element].begin(), vertices[element].end());
      for (const auto &element2 : vertices[element]) {
        globalTolocalCEM[i].insert({element2, j++});
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "======Finished with " << duration.count()
            << " ms======" << std::endl;
}

void System::formAUX() {
  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "======Phase III: Construct the Auxiliary space======"
            << std::endl;
  MKL_INT *rows_start, *rows_end, *col_index;
  MKL_INT rows, cols;
  sparse_index_base_t indexing;
  mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  int i = 0, j = 0;

  char which = 'S';
  MKL_INT pm[128];

  mkl_sparse_ee_init(pm);
  pm[7] = 1;
  pm[8] = 1;

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

  // for (i = 0; i < nparts; ++i) {
  //   Ai_col_index[i].reserve(nvtxs);
  //   Ai_row_index[i].reserve(nvtxs);
  //   Ai_values[i].reserve(nvtxs);
  //   Si_col_index[i].reserve(nvtxs);
  //   Si_row_index[i].reserve(nvtxs);
  //   Si_values[i].reserve(nvtxs);
  // }

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

  int k;

  eigenvalue.resize(nparts);
  for (i = 0; i < nparts; ++i) {
    eigenvalue[i].resize(k0);
  }

  eigenvector.resize(nparts);
  for (i = 0; i < nparts; ++i) {
    eigenvector[i].resize(k0 * count[i]);
  }

  double res[nparts];

  matrix_descr descr;

  descr.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
  descr.diag = SPARSE_DIAG_NON_UNIT;
  descr.mode = SPARSE_FILL_MODE_UPPER;

#pragma omp parallel for
  for (i = 0; i < nparts; ++i) {
    //   for (j = 0; j < Ai_values[i].size(); ++j) {
    //     std::cout<< Ai_col_index[i][j] << "  ";
    //   }
    //   std::cout << std::endl;
    //   for(j = 0; j < 11; ++j) {
    //     std::cout << Ai_row_index[i][j] << "  ";
    //   }
    //   std::cout << std::endl;
    //   for(j = 0; j < 11; ++j) {
    //     std::cout << Ai_values[i][j] << "  ";
    //   }
    //   std::cout << std::endl;

    //   for (j = 0; j < Si_values[i].size(); ++j) {
    //     std::cout<< Si_col_index[i][j] << "  ";
    //   }
    //   std::cout << std::endl;
    //   for (j = 0; j < Si_values[i].size(); ++j) {
    //     std::cout << Si_row_index[i][j] << "  ";
    //   }
    //   std::cout << std::endl;
    //   for (j = 0; j < Si_values[i].size(); ++j) {
    //     std::cout << Si_values[i][j] << "  ";
    //   }
    //   std::cout << std::endl;

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
      std::cout << "======error in " << error << "===============" << std::endl;
    }
    if (k < k0) {
      std::cout << "===========Not find enough eigenvalues==========="
                << std::endl;
    }
    // std::cout << "part: " << i << " residual: " << res[i]
    //           << " Smallest eigenvalue: " << eigenvalue[i][0] << std::endl;

    mkl_sparse_destroy(Ai[i]);
    mkl_sparse_destroy(Si[i]);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;

  std::cout << "======Finish solving eigen problem in each coarse element with "
            << duration.count() << " ms======" << std::endl;
}

void System::formCEM() {
  MKL_INT *rows_start, *rows_end, *col_index;
  MKL_INT rows, cols;
  sparse_index_base_t indexing;
  mkl_sparse_d_export_csr(matL, &indexing, &rows, &cols, &rows_start, &rows_end,
                          &col_index, &val);
  int i = 0, j = 0, k = 0;

  std::vector<double> sData(nvtxs, 0.0);
  for (i = 0; i < nvtxs; ++i) {
    for (j = rows_start[i]; j < rows_end[i]; ++j) {
      if (col_index[j] == i) {
        sData[i] += val[j] / cStar / cStar;
      } else {
        sData[i] += val[j] / cStar / cStar / 2;
      }
    }
  }

  std::vector<std::vector<double>> sMatrix;
  sMatrix.resize(nparts);
  for (i = 0; i < nparts; ++i) {
    sMatrix[i].resize(vertices[i].size() * vertices[i].size());
    for (const auto &element1 : vertices[i]) {
      for (const auto &element2 : vertices[i]) {
        sMatrix[i][globalTolocal[element1] * vertices[i].size() +
                   globalTolocal[element2]] = 0.0;
        for (j = 0; j < k0; ++j) {
          sMatrix[i][globalTolocal[element1] * vertices[i].size() +
                     globalTolocal[element2]] +=
              eigenvector[i][j * vertices[i].size() + globalTolocal[element1]] *
              eigenvector[i][j * vertices[i].size() + globalTolocal[element2]] *
              sData[element1] * sData[element2];
        }
      }
    }
  }

  for (i = 0; i < nparts; ++i) {
    std::vector<MKL_INT> Ai_col_index(
        verticesCEM[i].size() * verticesCEM[i].size() / overlapping[i].size(),
        0);
    std::vector<MKL_INT> Ai_row_index(
        verticesCEM[i].size() * verticesCEM[i].size() / overlapping[i].size(),
        0);
    std::vector<double> Ai_values(
        verticesCEM[i].size() * verticesCEM[i].size() / overlapping[i].size(),
        0);
    int index1 = 0;
    int index2 = 0;
    int index3 = 0;

    for (const auto &element : verticesCEM[i]) {
      for (j = rows_start[element]; j < rows_end[element]; ++j) {
        if (verticesCEM[i].count(col_index[j]) == 1) {
          if (col_index[j] < element) {
            Ai_row_index[index1++] = (globalTolocalCEM[i][element]);
            Ai_col_index[index2++] = (globalTolocalCEM[i][element]);
            Ai_values[index3++] = (val[j]);
          } else if (col_index[j] > element) {
            Ai_row_index[index1++] = (globalTolocalCEM[i][element]);
            Ai_col_index[index2++] = (globalTolocalCEM[i][element]);
            Ai_values[index3++] = (val[j]);
            Ai_row_index[index1++] = (globalTolocalCEM[i][element]);
            Ai_col_index[index2++] = (globalTolocalCEM[i][col_index[j]]);
            Ai_values[index3++] = (-val[j]);
          } else {
            Ai_row_index[index1++] = (globalTolocalCEM[i][element]);
            Ai_col_index[index2++] = (globalTolocalCEM[i][element]);
            Ai_values[index3++] = (val[j]);
          }
        }
      }
    }

    for (const auto &element : overlapping[i]) {
      for (const auto &element1 : vertices[element]) {
        for (const auto &element2 : vertices[element]) {
          if (globalTolocalCEM[i][element1] <= globalTolocalCEM[i][element2]) {
            Ai_row_index[index1++] = globalTolocalCEM[i][element1];
            Ai_col_index[index2++] = globalTolocalCEM[i][element2];
            Ai_values[index3++] =
                sMatrix[element]
                       [globalTolocal[element1] * vertices[element].size() +
                        globalTolocal[element2]];
          }
        }
      }
    }

    Ai_row_index.resize(index1);
    Ai_col_index.resize(index2);
    Ai_values.resize(index3);

    std::vector<double> rhs(verticesCEM[i].size() * k0, 0.0);
    for (j = 0; j < k0; ++j) {
      for (const auto &element : vertices[i]) {
        rhs[j * verticesCEM[i].size() + globalTolocalCEM[i][element]] =
            sData[globalTolocalCEM[i][element]] *
            eigenvector[i][j * vertices[i].size() + globalTolocal[element]];
      }
    }

    std::cout << "size: " << index1 << " " << index2 << " " << index3
              << std::endl;
  }

  std::vector<sparse_matrix_t> AiCOO;
  std::vector<sparse_matrix_t> Ai;

  AiCOO.resize(nparts);
  Ai.resize(nparts);

  // #pragma omp parallel for
  //   for (i = 0; i < nparts; ++i) {
  //     mkl_sparse_d_create_coo(&AiCOO[i], indexing, count[i], count[i],
  //                             Ai_values[i].size(), Ai_row_index[i].data(),
  //                             Ai_col_index[i].data(), Ai_values[i].data());

  //     mkl_sparse_convert_csr(AiCOO[i], SPARSE_OPERATION_NON_TRANSPOSE,
  //     &Ai[i]);

  //     mkl_sparse_destroy(AiCOO[i]);

  //     mkl_sparse_destroy(Ai[i]);
  //   }
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
  k0 = 3;
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
  k0 = 3;
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
  k0 = 3;
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
  k0 = 3;
}
//     fprintf(fp, "%d ", part[i]);
//   }
//   fclose(fp);
// }