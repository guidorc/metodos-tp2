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
  /*
  MatrixXd read(string filename) {
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    while (!infile.eof()) {
      string line;
      getline(infile, line);

      int temp_cols = 0;
      stringstream stream(line);
      while(!stream.eof())
        stream >> buff[cols*rows+temp_cols++];

      if (temp_cols == 0)
        continue;

      if (cols == 0)
        cols = temp_cols;

      rows++;
    }

    infile.close();

    rows--;

    // Populate matrix with numbers.
    MatrixXd result(rows,cols);
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        result(i,j) = buff[ cols*i+j ];

    return result;
  }
  */
  Matrix<double , Dynamic, Dynamic, RowMajor> read(string filename) {
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

  vector<eigenPair> deflationMethod(const Matrix<double , Dynamic, Dynamic, RowMajor> &m, int iterations, double epsilon) {
    Matrix<double , Dynamic, Dynamic, RowMajor> A = m;
    vector<eigenPair> result;
    double a = 0;
    VectorXd v = VectorXd::Zero(A.rows());
    eigenPair p;
    for (int i = 0; i < m.rows(); i++) {
      cout << "calculating eigenvalue: " << to_string(i) << endl;
      Matrix<double , Dynamic, Dynamic, RowMajor> sub = (a * v * v.transpose());
      A = A - sub;
      p = powerIteration(A, iterations, epsilon);
      result.push_back(p);
      a = p.eigenvalue;
      vector<double> ev = p.eigenvector;
      v = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(ev.data(), ev.size());
    }
    return result;
  }
}// namespace MatrixOperator
