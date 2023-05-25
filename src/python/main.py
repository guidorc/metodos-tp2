import numpy as np
import utils
import ejecutar
import IO

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
        # C = np.cov(np.transpose(X_c))
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
    res = []
    for i in range(len(X)):
        res.append(utils.reconstruirImagen(Z[i], V, k))
        utils.plotImage(res[i])
    return res


def TDPCA(imagenes, k):
    # Calculo image covariance matrix
    G = utils.imageCovarianceMatrix(imagenes)
    # Calculo base de autovectores
    utils.write("covarianza_2dpca.txt")
    ejecutar.correrTp("covarianza_2dpca.txt")
    U = IO.leerMatriz("covarianza_2dpca_eigenVectors.csv")
    for i, A in enumerate(imagenes):
        # Calculo matriz de feature vectors
        Y = np.matmul(A, np.transpose(U[i]))



if __name__ == '__main__':
    # Leer caras
    imagenes = IO.cargarImagenes()
    PCA(imagenes, 100, False)



