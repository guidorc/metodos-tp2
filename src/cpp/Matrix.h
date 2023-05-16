//
// Created by Guido Rodriguez on 11/05/2023.
//

#ifndef METODOS_TP2_MATRIX_H
#define METODOS_TP2_MATRIX_H

#include <list>
#include <vector>
#include "Eigenpair.h"
#include "string"
#include "Eigen/Sparse"

using namespace std;
using namespace Eigen;

namespace MatrixOperator {
  SparseMatrix<double> read(string filename);
  eigenPair power_iteration(const Matrix<double, Dynamic, Dynamic, RowMajor> &X, unsigned iterations, double epsilon);
}



#endif //METODOS_TP2_MATRIX_H
