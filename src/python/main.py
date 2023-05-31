import numpy as np
import utils
import ejecutar
import IO
import plotter

def obtenerMatricesCovarianzayCorrelación(X, sufijo = ''):
    # Matriz de covarianza
    print("Calculando Matriz de Covarianza")
    C = utils.matrizDeCovarianza(X)
    print("Calculando Matriz de Correlación")
    R = utils.matrizDeCorrelación(C)
    # Exportarla para calcular autovalores y autovectores
    print("Escribiendo Matriz de Covarianza")
    IO.write(np.matrix(C), "covarianza" + sufijo + ".txt")
    print("Escribiendo Matriz de Correlación")
    IO.write(np.matrix(R), "correlacion" + sufijo + ".txt")


def proyectarPCA(V, X, k):
    Z = []
    for i in range(len(X)):
        Z.append(utils.proyectar(V, X[i], k))
    return Z

def reconstruirPCA(V, X, k, h, w):
    imagenes_reconstruidas = []
    for i in range(len(X)):
        imagenes_reconstruidas.append(utils.reconstruirImagen(X[i], V, k))
    imagenes_formateadas = utils.formatearImagenes(imagenes_reconstruidas, h, w)
    return imagenes_formateadas


def PCA(imagenes, k, calcularCovarianza = False):
    # -------- PCA -------- #
    X = utils.aplanarImagenes(imagenes)
    # Obtener componentes principales
    if calcularCovarianza:
        obtenerMatricesCovarianzayCorrelación(X)
        # Calcular autovalores y autovectores de matriz de covarianza
        print("Ejecutando Deflacion para Matriz de Covarianza")
        ejecutar.correrTp("covarianza")
    V = IO.leerMatriz("resultados/", "covarianza_eigenVectors.csv")
    # Obtener proyeccion de menor dimension
    Z = proyectarPCA(V, X, k)
    # Reconstruir imagenes
    _, h, w = imagenes.shape
    return reconstruirPCA(V, X, k, h, w), Z


def proyectarTDPCA(Y, k):
    # Y de a x b
    Z = Y[:k]
    return Z


def reconstruirTDPCA(M, U, k):
    h = len(M[0][0])
    w = len(U[0])
    U_t = np.transpose(U)
    imagenes_reconstruidas = []
    for Y in M:
        A = np.zeros((h, w))
        for i in range(k):
            y_i = Y[i]
            x_i = U_t[i]
            A += np.outer(y_i, x_i)
        imagenes_reconstruidas.append(A)
    for i, imagen in enumerate(imagenes_reconstruidas):
        IO.write(imagen, str(i) + '.pgm', 'resultados/caras/s1')
    return np.array(imagenes_reconstruidas)

def TDPCA(imagenes, k, calcularAutovectores=False):
    # Calculo image covariance matrix
    G = utils.imageCovarianceMatrix(imagenes)
    # Calculo matriz de correlacion
    R = utils.matrizDeCorrelacionTDPCA(G)
    # Calculo base de autovectores
    if calcularAutovectores:
        # Calcular autovectores de G
        IO.write(np.matrix(G), "covarianza_2dpca.txt")
        IO.write(np.matrix(G), "correlacion_2dpca.txt")
        ejecutar.correrTp("covarianza_2dpca")
    U = IO.leerMatriz("resultados/", "covarianza_2dpca_eigenVectors.csv")
    # Calcular feature vectors
    feature_matrix = []  # de n x a x b
    for _, A in enumerate(imagenes):
        # Calculo matriz de feature vectors para A de a x b
        Y = []  # Y de a x b
        for i in range(len(U)):
            Y.append(np.matmul(A, U[i]))
        feature_matrix.append(Y)
    # Obtener proyeccion de menor dimension
    for imagen in feature_matrix:
        proyectarTDPCA(imagen, k)
    # Reconstruir imagenes
    return reconstruirTDPCA(feature_matrix, U, k), feature_matrix[:k]


if __name__ == '__main__':
    # Leer caras
    imagenes = IO.cargarImagenes()

    # -------- PCA -------- #
    imagenes_pca, z_pca = PCA(imagenes, 100, False)
    # obtenerMatricesCovarianzayCorrelación(z_pca, "_pca")
    # plotter.imprimirImagenes(imagenes_pca)

    # -------- 2DPCA -------- #
    imagenes_tdpca, z_tdpca = TDPCA(imagenes, 10, False)
    # obtenerMatricesCovarianzayCorrelación(z_tdpca, "_tdpca")
    # plotter.imprimirImagenes(imagenes_tdpca)

    # -------- EXPERIMENTACION -------- #
    # plotter.graficarAutovalores("covarianza_eigenValues.csv")
    # plotter.graficarCorrelacion("correlacion.txt")

