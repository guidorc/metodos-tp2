#include "IO.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace std;

namespace IO {
  bool compareByEigenValue(const eigenPair &ep1, const eigenPair &ep2) {
    return ep1.eigenvalue > ep2.eigenvalue;
  }

  void writeEigenVectors(ofstream &file, vector<eigenPair> &results) {
    vector<vector<double>> transposed(results[0].eigenvector.size());
    for (eigenPair pair: results) {
      int row = 0;
      for (double v_i: pair.eigenvector) {
        // we save the eigenvectors as columns
        transposed[row].push_back(v_i);
        row++;
      }
    }

    for (double i = 0; i < transposed.size(); i++) {
      // column names
      file << "v_" << i + 1 << ", ";
    }
    file << "\n";

    for (double i = 0; i < transposed.size(); i++) {
      for (double v_i: transposed[i]) {
        file << v_i << ", ";
      }
      file << "\n";
    }
    file.close();
  }

  void writeResults(vector<eigenPair> &results, string filename) {
    ostringstream streamObj;
    streamObj << fixed;
    streamObj << setprecision(4);
    ofstream eigenValues;
    ofstream eigenVectors;
    eigenValues.open("./resultados/" + filename + "_eigenValues.csv");
    if (eigenValues.fail()) {
      cout << "unable to create eigenvalues for: " << filename << endl;
    }

    eigenVectors.open("./resultados/" + filename + "_eigenVectors.csv");
    if (eigenVectors.fail()) {
      cout << "unable to create eigenvectors for: " << filename << endl;
    }

    // ordenar autovalores de mayor a menor
    std::sort(results.begin(), results.end(), compareByEigenValue);

    // guardar autovalores
    eigenValues << "eigenValues,\n";
    for (eigenPair pair: results) {
      eigenValues << pair.eigenvalue << ",\n";
    }
    eigenValues.close();

    // guardar autovectores
    writeEigenVectors(eigenVectors, results);
  }
}// namespace IO