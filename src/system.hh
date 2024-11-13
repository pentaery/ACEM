#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_types.h"
#include <cmath>
#include <vector>

class System {
  sparse_matrix_t A;
  std::vector<double> rhs;
  std::vector<double> sol;
  std::vector<MKL_INT> row_indx;
  std::vector<MKL_INT> col_indx;
  std::vector<double> values;
  sparse_index_base_t indexing;
  MKL_INT rows, cols;
  MKL_INT *rows_start, *rows_end, *col_index;
  double *val;
  MKL_INT size;

  MKL_INT perm[64], iparm[64];
  void *pt[64];

public:
  void formRHS();
  void formA();
  void solve();
  System();
  System(MKL_INT size);
};
