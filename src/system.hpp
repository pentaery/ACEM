#include <Eigen/Sparse>
#include <cmath>
#include <vector>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;
typedef Eigen::SparseVector<double, Eigen::RowMajor> SpVec;
typedef Eigen::Triplet<double> T;

void formSystem(SpMat &A, SpVec &f, int size);
