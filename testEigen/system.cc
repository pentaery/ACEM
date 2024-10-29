#include "system.hh"
#include <math.h>

void formSystem(SpMat &A, SpVec &f, int size) {
  std::vector<T> tripletList;
  tripletList.reserve(5 * size);
  double lx = 1.0;
  double ly = 1.0;
  double hx = lx / (size + 1);
  double hy = ly / (size + 1);
  int row = 0, col = 0;
  for (row = 0; row < size; ++row) {
    for (col = 0; col < size; ++col) {
      tripletList.push_back(
          T(row * size + col, row * size + col, 4.0 / hx / hy));
      if (row > 0) {
        tripletList.push_back(
            T(row * size + col, (row - 1) * size + col, -1.0 / hx / hy));
      }
      if (row < size - 1) {
        tripletList.push_back(
            T(row * size + col, (row + 1) * size + col, -1.0 / hx / hy));
      }
      if (col > 0) {
        tripletList.push_back(
            T(row * size + col, row * size + col - 1, -1.0 / hx / hy));
      }
      if (col < size - 1) {
        tripletList.push_back(
            T(row * size + col, row * size + col + 1, -1.0 / hx / hy));
      }
      f.insert(row * size + col) = 2 * M_PI * M_PI *
                                   sin(M_PI * (col + 1) * hx) *
                                   sin(M_PI * (row + 1) * hy);
    }
  }
  A.setFromTriplets(tripletList.begin(), tripletList.end());
}
