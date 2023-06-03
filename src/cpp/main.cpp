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
  string filename = "./matrices/" + input + ".txt";
  Matrix<double , Dynamic, Dynamic, RowMajor> M = readMatrixFromFile(filename);

  int k = M.rows();
  if (argc > 4) {
    k = atof(argv[4]);
  }

  // Calculo de autovalores y autovectores
  vector<eigenPair> res = deflationMethod(M, iterations, tolerance, k);

  // Escritura de resultados
  writeResults(res, input);

  return 0;
}
