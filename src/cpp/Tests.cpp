//
// Created by clinux01 on 30/06/23.
//

#include "Tests.h"

using namespace std;
using namespace chrono;
using namespace MatrixOperator;

void testConocida() {
  std::cout << "Test Matriz Conocida" << endl;
  Matrix<double, Dynamic, Dynamic, RowMajor> M = readMatrixFromFile("./matrices/tests/conocida.txt");
  vector<eigenPair> autovalores = deflationMethod(M, 1000, 0.001, M.rows());
  vector<double> autovalores_esperados {4, 2, 1};

  for(int i = 0; i < autovalores.size(); i++){
    std::cout << "Autovalor numero " << i << endl;
    double esperado = autovalores_esperados[i];
    double encontrado = autovalores[i].eigenvalue;
    std::cout << "Valor esperado: " << esperado << endl;
    std::cout << "Valor encontrado:" << encontrado << endl;
  };
};

void testSDP(){
  std::cout << "Test Matriz Simetrica definida positiva" << endl;
  Matrix<double , Dynamic, Dynamic, RowMajor> sdp = readMatrixFromFile("./matrices/tests/sdp.txt");
  vector<eigenPair> autovalores = deflationMethod(sdp, 1000, 0.001, sdp.rows());
  EigenSolver<MatrixXd> eigensolver(sdp);

  for(int i = 0; i < autovalores.size(); i++){
    std::cout << "Autovalor numero " << i << endl;
    double esperado = eigensolver.eigenvalues().col(0)[i].real();
    double encontrado = autovalores[i].eigenvalue;
    std::cout << "Valor esperado: " << esperado << endl;
    std::cout << "Valor encontrado:" << encontrado << endl;
  };
};

void testAutovaloresCercanosACero(){
  std::cout << "Test Matriz Con Autovalores Cercanos a 0" << endl;
  Matrix<double , Dynamic, Dynamic, RowMajor> T = readMatrixFromFile("./matrices/tests/triangular.txt");

  vector<eigenPair> autovalores = deflationMethod(T, 1000, 0.001, T.rows());
  vector<double> autovalores_esperados {1, 0.1, 0.001, 0.0001, 0.00001, 0.000001};

  for(int i = 0; i < autovalores.size(); i++){
    std::cout << "Autovalor numero " << i << endl;
    double esperado = autovalores_esperados[i];
    double encontrado = autovalores[i].eigenvalue;
    std::cout << "Valor esperado: " << esperado << endl;
    std::cout << "Valor encontrado:" << encontrado << endl;
  };
};

void runTests() {
  testConocida();
  testSDP();
  testAutovaloresCercanosACero();
};
