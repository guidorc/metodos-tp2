import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def write(m, filename, path='matrices/'):
    with open(path + filename, 'wb') as f:
        for line in m:
            np.savetxt(f, line, fmt='%.6f')


def writeTimes(data, filename):
    with open(r'resultados/ejercicio_4/' + filename, 'w') as fp:
        for time in data:
            # write each item on a new line
            fp.write("%.3f\n" % time)


def cargarImagenes():
    paths = []
    imgs = []
    for i in range(1, 42):
        directorio = "matrices/caras/s" + str(i)
        for path in sorted(list(Path(directorio).rglob('*.pgm'))):
            paths.append(path)
            image = (plt.imread(path)[::2, ::2]/255)
            imgs.append(image)
    result = np.stack(imgs)
    return result

def leerMatriz(path, filename, cols=None):
    # LEE LAS MATRICES POR COLUMNAS
    matrix = open(path + filename)
    if cols:
        return np.genfromtxt(matrix, delimiter=",", names=True, dtype=None, unpack=True, usecols=range(cols))
    else:
        return np.genfromtxt(matrix, delimiter=",", names=True, dtype=None, unpack=True)[:-1]


def leerMatricesCorrelacion(filenames):
    data = []
    for i, filename in enumerate(filenames):
        data.append(leerMatrizCorrelacion("matrices/", filename + ".txt"))
    return data


def leerMatrizCorrelacion(path, filename):
    matrix = open(path + filename)
    return np.loadtxt(matrix, delimiter=' ')