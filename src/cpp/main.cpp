#include <iostream>

#include "Matrix.h"
#include "IO.h"

using namespace MatrixOperator;
using namespace IO;
using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cerr << "Formato de entrada: " << argv[0]
         << "<archivo> <iteraciones> <tolerancia>" << endl;
    return 1;
  }

  // Lectura de parametros:
  string input = argv[1];
  int iterations = atof(argv[2]);
  double tolerance = atof(argv[3]);

  // Lectura de matriz
  SparseMatrix<double> M = read("./matrices/" + input + ".txt");

  // Calculo de autovalores y autovectores
  vector<eigenPair> res = deflationMethod(M, iterations, tolerance);

  // Escritura de resultados
  writeResults(res, input);

  return 0;
}
