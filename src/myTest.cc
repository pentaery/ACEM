#include <iostream>
#include <vector>

int main() {
  std::vector<int> v;
  v[0] = 1;
  v.reserve(10);
  v.push_back(1);
  v.push_back(2);
  std::cout << "v.size() = " << v.size() << std::endl;
}