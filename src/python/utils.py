import numpy as np


# print(np.cov(matrizA))
def matrizDeCovarianza(m):
    return np.dot(1 / (len(m) - 1), np.matmul(m.transpose(), m))


# Matriz de m x n
# matrizA.mean(axis=0)
def promedioDimensionPixel(matrizX):
    m = len(matrizX)
    n = len(matrizX[0])
    res = []
    for i in range(0, n):
        res.append(promedioDimensionPixelAux(matrizX, i))
    return res


# Matriz de m x n
def promedioDimensionPixelAux(matrizX, columnaJ):
    # Yo lo que quiero es el promedio de la columna J de X
    m = len(matrizX)
    res = 0
    for fila in matrizX:
        res += fila[columnaJ]
    return res / m


def write(m, filename):
    with open('matrices/' + filename, 'wb') as f:
        for line in m:
            np.savetxt(f, line, fmt='%.2f')
