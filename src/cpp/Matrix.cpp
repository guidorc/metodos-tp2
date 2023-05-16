//
// Created by Guido Rodriguez on 11/05/2023.
//

#include "Matrix.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace MatrixOperator;

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
  SparseMatrix<double> read(string filename) {
    ifstream file(filename.c_str());
    vector<vector<double>> res;
    string line, temp;
    // build matrix
    vector<Triplet<double>> inputReader;
    int rowNumber = 0;
    while (getline(file, line)) {
      list<string> linkList = split(line, ' ');
      auto current = linkList.begin();
      auto last = linkList.end();
      int columnNumber = 0;
      while (current != last) {
        const string &ref = *current;
        inputReader.push_back(Triplet(rowNumber, columnNumber, stod(ref)));
        current = std::next(current, 1);
        columnNumber++;
      }
      rowNumber++;
    }
    SparseMatrix<double> result(rowNumber, rowNumber);
    result.setFromTriplets(inputReader.begin(), inputReader.end());
    return result;
  }

  eigenPair power_iteration(const Matrix<double, Dynamic, Dynamic, RowMajor> &m, unsigned int iterations, double epsilon) {
    VectorXd previousVector = VectorXd::Random(m.cols());

    for (unsigned int i = 0; i < iterations; i++) {
      VectorXd multipliedVector = m * previousVector;
      multipliedVector = multipliedVector / multipliedVector.norm();
      double cos_angle = multipliedVector.transpose() * previousVector;
      previousVector = multipliedVector;
      if ((1 - epsilon) < cos_angle && cos_angle <= 1) {
        break;
      }
    }

    eigenPair eigenPair;

    eigenPair.eigenvalue = previousVector.transpose() * m * previousVector;
    for (VectorXd::iterator it = previousVector.begin(); it != previousVector.end(); it++) {
      eigenPair.eigenvector.push_back(*it);
    }
    return eigenPair;
  }

    vector<eigenPair> deflationMethod(const Matrix<double, Dynamic, Dynamic, RowMajor> &m, int iterations, double epsilon) {
        Matrix<double, Dynamic, Dynamic, RowMajor> A = m;
        vector<eigenPair> result;
        double a = 0;
        VectorXd v = VectorXd::Zero(A.rows());
        eigenPair p;
        for (int i = 0; i < m.rows(); i++)
        {
            A = A - (a * v * v.transpose());
            p = power_iteration(A, iterations, epsilon);
            result.push_back(p);
            a = p.eigenvalue;
            vector<double> ev = p.eigenvector;
            v = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(ev.data(), ev.size());
            cout << "Found eigenvalue number: " << i << ".  Value: " <<  p.eigenvalue << endl;
        }
        return result;
    }
} // namespace MatrixOperator

