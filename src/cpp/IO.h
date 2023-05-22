//
// Created by Guido Rodriguez on 11/05/2023.
//

#ifndef METODOS_TP2_IO_H
#define METODOS_TP2_IO_H

#include <string>
#include <vector>
#include "Matrix.h"
#include "Eigenpair.h"

using namespace std;

namespace IO {
  void writeResults(vector<eigenPair> &pairs, string file);
  void writeEigenVectors(ofstream &file, vector<eigenPair> &results);
  bool compareByEigenValue(const eigenPair & ep1, const eigenPair & ep2);
} // namespace IO

#endif //METODOS_TP2_IO_H
