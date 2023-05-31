import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def write(m, filename, path='matrices/'):
    with open(path + filename, 'wb') as f:
        for line in m:
            np.savetxt(f, line, fmt='%.6f')


def cargarImagenes():
    paths = []
    imgs = []
    for path in sorted(list(Path('matrices/caras/s1').rglob('*.pgm'))):
        paths.append(path)
        image = (plt.imread(path)[::2, ::2]/255)
        imgs.append(image)
    result = np.stack(imgs)
    return result

def leerMatriz(path, filename):
    matrix = open(path + filename)
    # return pd.read_csv(matrix, delimiter=",", dtype=None).T.values[:-1]
    return np.genfromtxt(matrix, delimiter=",", names=True, dtype=None, unpack=True)[:-1]

def leerMatrizCorrelacion(path, filename):
    matrix = open(path + filename)
    # return pd.read_csv(matrix, delimiter=",", dtype=None).T.values[:-1]
    input = np.loadtxt(matrix, dtype='i', delimiter=' ')
    return input
    # return np.genfromtxt(matrix, delimiter=" ", names=True, dtype=None, unpack=True)