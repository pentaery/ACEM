#include "mkl_spblas.h"
#include <iostream>
#include <vector>

int main() {
  sparse_matrix_t test;
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  MKL_INT rows = 2, cols = 2;
  // MKL_INT rows_start[3] = {0, 1};
  std::vector<MKL_INT> rows_start = {0, 1};
  std::vector<MKL_INT> rows_end = {1, 2};
  std::vector<MKL_INT> col_index = {0, 1};
  // MKL_INT rows_end[3] = {1, 2};
  // MKL_INT col_index[2] = {0, 1};
  double vall[2] = {1.0, 1.0};

  std::cout << vall << " " << rows_start.data() << std::endl;

  mkl_sparse_d_create_csr(&test, indexing, rows, cols, rows_start.data(), rows_end.data(),
                          col_index.data(), vall);

  MKL_INT rows1, cols1;
  MKL_INT *rows_start1, *rows_end1, *col_index1;
  double *vall1;

  mkl_sparse_d_export_csr(test, &indexing, &rows1, &cols1, &rows_start1,
                          &rows_end1, &col_index1, &vall1);

  std::cout << vall1 << " " << rows_start1 << std::endl;
}