#include <Eigen/Sparse>
#include <cmath>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::Triplet<double> T;

void formSystem(SpMat &A, SpVec &f, int size);
