#include <iostream>

#include "Matrix.h"

using namespace MatrixOperator;
using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cerr << "Formato de entrada: " << argv[0]
         << "<archivo> <iteraciones> <tolerancia>" << endl;
    return 1;
  }

  // lectura parametros:
  string input = argv[1];
  int iterations = atof(argv[2]);
  double tolerance = atof(argv[3]);

  SparseMatrix<double> M = read("./matrices/" + input + ".txt");

  vector<eigenPair> res = deflationMethod(M, iterations, tolerance);

  return 0;
}
