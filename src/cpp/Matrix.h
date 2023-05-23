//
// Created by Guido Rodriguez on 11/05/2023.
//

#ifndef METODOS_TP2_MATRIX_H
#define METODOS_TP2_MATRIX_H
#define MAXBUFSIZE  ((int) 1e6)

#include <list>
#include <vector>
#include "Eigenpair.h"
#include "string"
#include "./Eigen/Dense"

using namespace std;
using namespace Eigen;

namespace MatrixOperator {
  MatrixXd read(string filename);
  eigenPair powerIteration(const MatrixXd &X, unsigned iterations, double epsilon);
  vector<eigenPair> deflationMethod(const MatrixXd &m, int iterations, double epsilon);
}



#endif //METODOS_TP2_MATRIX_H
