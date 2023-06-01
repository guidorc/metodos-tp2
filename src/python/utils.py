import math
import numpy as np


def centrarMatriz(X):
    # promedio dimension-pixel
    muj = X.mean(axis=0)
    # matriz centrada
    X_c = []
    for x_i in X:
        X_c.append(x_i - muj)
    return X_c

def calcularCovarianza(m):
    return np.dot(1 / (len(m) - 1), np.matmul(m.transpose(), m))

def matrizDeCorrelación(C):
    rows, cols = C.shape
    R = np.zeros((rows, cols))

    for i in range(cols):
        for j in range(cols):
            R[i][j] = C[i][j] / (math.sqrt(C[i][i]*C[j][j]))
    return R

def matrizDeCovarianza(X):
    # centrar matriz
    X_c = centrarMatriz(X)
    # Matriz de covarianza
    C = calcularCovarianza(np.array(X_c))
    return C

def aplanarImagenes(imagenes):
    X = []
    for imagen in imagenes:
        X.append(imagen.flatten())
    return np.array(X)


def proyectar(V, x_i, k):
    z_i = []
    for i in range(k):
        z_i.append(np.matmul(np.transpose(V[i]), x_i))
    return np.array(z_i)

def reconstruirImagen(x, V, k):
    z_x = np.matmul(np.transpose(V), x)
    x = np.dot(z_x[0], V[0])
    for i in range(1, k):
        x += np.dot(z_x[i], V[i])
    return x

def formatearImagenes(imagenes, h, w):
    # imagenes viene aplanada
    res = []
    for imagen in imagenes:
        imagen_formateada = np.reshape(imagen, (h, w))
        res.append(imagen_formateada)
    return np.array(res)
def imagenPromedio(imagenes):
    P = np.zeros(np.shape(imagenes[0]))
    for i in range(len(imagenes)):
        m_i = imagenes[i]
        P += m_i
    return np.dot(1 / (len(imagenes)), P)

def imageCovarianceMatrix(imagenes):
    n = len(imagenes)
    P = imagenPromedio(imagenes)
    # inicializar G con 0s
    rows, cols = np.shape(imagenes[0])
    G = np.zeros((cols, cols))
    for i in range(n):
        A_i = imagenes[i]
        X = A_i - P
        G += np.matmul(np.transpose(X), X)
    return np.dot(1/n, G)


def matrizDeCorrelacionTDPCA(C):
    rows, cols = C.shape
    R = np.zeros((rows, cols))

    for i in range(cols):
        for j in range(cols):
            R[i][j] = C[i][j] / (math.sqrt(C[i][i]*C[j][j]))
    return R