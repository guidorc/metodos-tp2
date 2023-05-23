import numpy as np
import utils
import ejecutar
import IO

def PCA(imagenes):
    # Construir matriz de imagenes
    X = []
    for imagen in imagenes:
        X.append(imagen.flatten())
    X = np.array(X)
    # centrar matriz
    X_c = utils.centrarMatriz(X)
    # Matriz de covarianza
    # CONSULTAR SI ES LO MISMO HACER X^t * X o X*X^t
    C = utils.matrizDeCovarianza(np.array(X_c))
    # Exportarla para calcular autovalores y autovectores
    utils.write(np.matrix(C), "covarianza.txt")
    # Calcular autovalores y autovectores de matriz de covarianza
    ejecutar.correrTp("covarianza")
    # Leer matriz de autovectores
    V = IO.leerMatriz("covarianza_eigenVectors.csv")
    Z = []
    for i in range(len(X)):
        Z.append(utils.proyectar(V, X[i], 2))
    return Z

if __name__ == '__main__':
    # Leer caras
    imagenes = IO.cargarImagenes()
    PCA(imagenes)



