#include "mkl_spblas.h"
#include <cmath>
#include <vector>
#include "mkl.h"

void formRHS(std::vector<double> &vec, MKL_INT size);
void formA(sparse_matrix_t &A, MKL_INT size);