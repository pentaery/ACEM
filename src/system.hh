#include "mkl_spblas.h"
#include <cmath>
#include <vector>
#include "mkl.h"
#define precision float

void formRHS(std::vector<precision> &vec, int size);
void formA(sparse_matrix_t &A, int size);