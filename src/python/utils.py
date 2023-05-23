import numpy as np

def centrarMatriz(X):
    # promedio dimension-pixel
    muj = X.mean(axis=0)
    # matriz centrada
    X_c = []
    for x_i in X:
        X_c.append(x_i - muj)
    return X_c


def matrizDeCovarianza(m):
    return np.dot(1 / (len(m) - 1), np.matmul(m, m.transpose()))


def write(m, filename):
    with open('matrices/' + filename, 'wb') as f:
        for line in m:
            np.savetxt(f, line, fmt='%.2f')


def proyectar(V, x_i, k=2):
    z_i = []
    for i in range(k):
        z_i.append(np.matmul(V[i], x_i))
    return z_i
