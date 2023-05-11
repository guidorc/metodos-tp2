#include <iostream>

#include "Matrix.h"

using namespace MatrixOperator;
using namespace std;

int main(int argc, char *argv[]) {
  string input = "./matrices/diagonal.txt";

  SparseMatrix<double> M = read(input);

  eigenPair res = power_iteration(M, 10, 0.0001);

  cout << "prueba" << endl;

  return 0;
}
