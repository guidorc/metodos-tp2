//
// Created by Guido Rodriguez on 11/05/2023.
//

#ifndef METODOS_TP2_MATRIX_H
#define METODOS_TP2_MATRIX_H
#define MAXBUFSIZE  ((int) 1e8)

#include <list>
#include <vector>
#include "Eigenpair.h"
#include "string"
#include "./Eigen/Sparse"

using namespace std;
using namespace Eigen;

namespace MatrixOperator {
  Matrix<double , Dynamic, Dynamic, RowMajor> read(string filename);
  eigenPair powerIteration(const Matrix<double , Dynamic, Dynamic, RowMajor> &X, unsigned iterations, double epsilon);
  vector<eigenPair> deflationMethod(const Matrix<double , Dynamic, Dynamic, RowMajor> &m, int iterations, double epsilon);
}



#endif //METODOS_TP2_MATRIX_H
