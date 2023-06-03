//
// Created by Guido Rodriguez on 11/05/2023.
//

#include "Matrix.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace Eigen;

list<string> split(string originalString, char delim) {
  list<string> output;
  string current;
  stringstream stream(originalString);

  while (getline(stream, current, delim)) {
    output.push_back(current);
  }

  return output;
}

namespace MatrixOperator {
  Matrix<double, Dynamic, Dynamic, RowMajor> readMatrixFromFile(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
      std::cerr << "Error opening file: " << filename << std::endl;
      // Return an empty matrix or handle the error in an appropriate way
      return MatrixXd();
    }

    std::vector<std::vector<double>> matrixData;

    std::string line;
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      std::vector<double> row;

      double value;
      while (iss >> value) {
        row.push_back(value);
      }

      if (!row.empty()) {
        matrixData.push_back(row);
      }
    }

    infile.close();

    if (matrixData.empty()) {
      std::cerr << "Empty matrix in file: " << filename << std::endl;
      // Return an empty matrix or handle the error in an appropriate way
      return MatrixXd();
    }

    int rows = matrixData.size();
    int cols = matrixData[0].size();

    Matrix<double, Dynamic, Dynamic, RowMajor> matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        matrix(i, j) = matrixData[i][j];
      }
    }

    return matrix;
  }

  eigenPair powerIteration(const Matrix<double , Dynamic, Dynamic, RowMajor> &A, unsigned int iterations, double epsilon) {
    VectorXd previousVector = VectorXd::Random(A.cols());

    for (unsigned int i = 0; i < iterations; i++) {
      VectorXd currentVector = A * previousVector;
      currentVector = currentVector / currentVector.norm();
      // Criterio de parada usando angulo entre los vectores
      double cos_angle = currentVector.transpose() * previousVector;
      if ((1 - epsilon) < cos_angle && cos_angle <= 1) {
        break;
      }
      previousVector = currentVector;
    }

    eigenPair result;

    result.eigenvalue = previousVector.transpose() * A * previousVector;
    for (VectorXd::iterator it = previousVector.begin(); it != previousVector.end(); it++) {
      result.eigenvector.push_back(*it);
    }
    return result;
  }

  std::vector<eigenPair> deflationMethod(const Matrix<double , Dynamic, Dynamic, RowMajor> &m, int iterations, double epsilon, int k) {
    Matrix<double , Dynamic, Dynamic, RowMajor> A = m;
    std::vector<eigenPair> result;
    double a = 0;
    VectorXd v = VectorXd::Zero(A.rows());
    eigenPair p;
    for (int i = 0; i < k; i++) {
      cout << "calculating eigenvalue: " << to_string(i) << endl;
      Matrix<double , Dynamic, Dynamic, RowMajor> sub = (a * v * v.transpose());
      A = A - sub;
      p = powerIteration(A, iterations, epsilon);
      result.push_back(p);
      a = p.eigenvalue;
      std::vector<double> ev = p.eigenvector;
      v = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(ev.data(), ev.size());
    }
    return result;
  }
}// namespace MatrixOperator
