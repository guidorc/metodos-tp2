import numpy as np
import matplotlib.pyplot as plt

def centrarMatriz(X):
    # promedio dimension-pixel
    muj = X.mean(axis=0)
    # matriz centrada
    X_c = []
    for x_i in X:
        X_c.append(x_i - muj)
    return X_c


def matrizDeCovarianza(m):
    return np.dot(1 / (len(m) - 1), np.matmul(m.transpose(), m))


def write(m, filename):
    with open('matrices/' + filename, 'wb') as f:
        for line in m:
            np.savetxt(f, line, fmt='%.6f')


def proyectar(V, x_i, k):
    z_i = []
    for i in range(k):
        z_i.append(np.matmul(V[i], x_i))
    return z_i

def reconstruirImagen(z_x, V, k):
    x = [z_x]
    for i in range(1, k):
       x += np.matmul(z_x[i], V[i])
    return x

def imagenPromedio(imagenes):
    return imagenes[0]

def imageCovarianceMatrix(imagenes):
    n = len(imagenes)
    P = imagenPromedio(imagenes)
    G = []
    for i in range(len(imagenes)):
        A_i = imagenes[i]
        X = A_i - P
        G += np.matmul(np.transpose(X), X)
    return np.dot(1/n, G)

def plotImage(image):
    plt.imshow(image, cmap='gray')

