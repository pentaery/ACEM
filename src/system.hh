#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_types.h"
#include <cmath>
#include <vector>

class System {
  sparse_matrix_t matA;
  sparse_matrix_t matL;
  std::vector<double> vecRHS;
  std::vector<double> vecSOL;
  std::vector<double> matM;
  sparse_index_base_t indexing;
  MKL_INT rows, cols;

  double *val;
  MKL_INT size;
  MKL_INT nvtxs;

  MKL_INT perm[64], iparm[64];
  void *pt[64];

public:
  void getL();
  void getM();
  void formRHS();
  void formA();
  void solve();
  void graphPartition();
  System();
  System(MKL_INT size);
};
