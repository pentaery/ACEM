#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_types.h"
#include <cmath>
#include <metis.h>
#include <set>
#include <unordered_set>
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

  idx_t nparts;
  idx_t *part;

  std::vector<std::set<idx_t>> vertices;
  std::vector<std::unordered_set<idx_t>> neighbours;
  std::vector<std::unordered_set<idx_t>> overlapping;

  int overlap;

public:
  void getData();
  void formRHS();
  void formA();
  void solve();
  void graphPartition();
  void findNeighbours();
  void formAUX();
  System();
  System(MKL_INT size);
  System(MKL_INT size, idx_t nparts);
  System(MKL_INT size, idx_t nparts, int overlap);
};
