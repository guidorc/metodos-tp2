import numpy as np
import utils
import ejecutar
import imagenes

if __name__ == '__main__':
    # Leer caras
    imagenes = imagenes.cargarImagenes()
    # Construir matriz de imagenes
    X = []
    for imagen in imagenes:
        X.append(imagen.flatten())
    X = np.array(X)
    # promedio dimension-pixel
    muj = X.mean(axis=0)
    # matriz centrada
    X_c = []
    for x_i in X:
        X_c.append(x_i - muj)
    # Matriz de covarianza
    C = utils.matrizDeCovarianza(np.array(X_c))
    # Exportarla para calcular autovalores y autovectores
    # utils.write(np.matrix(C), "covarianza.txt")
    # Calcular autovalores y autovectores de matriz de covarianza
    # ejecutar.corerTp("covarianza")


