import numpy as np
import utils
import ejecutar
import IO
import plotter

def PCA(imagenes, k, calcularCovarianza = False):
    # Aplanar imagenes
    X = []
    for imagen in imagenes:
        X.append(imagen.flatten())
    X = np.array(X)
    # centrar matriz
    X_c = utils.centrarMatriz(X)
    if calcularCovarianza:
        # Matriz de covarianza
        print("Calculando Matriz de Covarianza")
        C = utils.matrizDeCovarianza(np.array(X_c))
        # Exportarla para calcular autovalores y autovectores
        print("Escribiendo Matriz de Covarianza")
        utils.write(np.matrix(C), "covarianza.txt")
        # Calcular autovalores y autovectores de matriz de covarianza
        print("Ejecutando Deflacion para Matriz de Covarianza")
        ejecutar.correrTp("covarianza")
    # Leer matriz de autovectores
    V = IO.leerMatriz("covarianza_eigenVectors.csv")
    # Proyectar imagenes
    Z = []
    for i in range(len(X)):
        Z.append(utils.proyectar(V, X[i], k))
    # Reconstruir imagenes
    # res = []
    # for i in range(len(X)):
    #    res.append(utils.reconstruirImagen(Z[i], V, k))
    #    utils.plotImage(res[i])
    # return res


def TDPCA(imagenes, k, calcularAutovectores):
    # Calculo image covariance matrix
    G = utils.imageCovarianceMatrix(imagenes)
    # Calculo base de autovectores
    if calcularAutovectores:
        # Calcular autovectores de G
        utils.write(np.matrix(G), "covarianza_2dpca.txt")
        ejecutar.correrTp("covarianza_2dpca")
    U = IO.leerMatriz("covarianza_2dpca_eigenVectors.csv")
    Y = []
    for _, A in enumerate(imagenes):
        # Calculo i-esimo feature vector
        y_i = []
        for i in range(k):
            y_i.append(np.matmul(A, U[i]))
        Y.append(y_i)
    # Reconstruir imagenes
    Res = np.zeros((len(Y[0][0]), len(U[0])))
    for i in range(k):
        y_i = Y[0][i]
        x_i = U[i]
        Res += np.matmul(y_i, np.transpose(x_i))


if __name__ == '__main__':
    # Leer caras
    imagenes = IO.cargarImagenes()
    # PCA(imagenes, 100, False)
    TDPCA(imagenes, 10, False)
    # plotter.graficarAutovalores("covarianza_eigenValues.csv")


# Este comentario es para ver si puedo pushear cosas.
