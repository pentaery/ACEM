#include "system.hh"
#include <cmath>

void formRHS(std::vector<precision> &vec, int size) {
  precision gridLength = 1.0 / 513;
  int row = 0, col = 0;
  for (row = 0; row < size; ++row) {
    for (col = 0; col < size; ++col) {
      vec[row * size + col] = M_PI * M_PI * sin(M_PI * (row + 1) * gridLength) * sin(M_PI * (col + 1) * gridLength);
    }
  }
}

