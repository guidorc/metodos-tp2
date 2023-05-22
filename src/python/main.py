import numpy as np
import utilidad

if __name__ == '__main__':
    matrizA = np.array([[1, 2], [3, 4]])
    matrizB = np.array([[1, 3], [3, 6]])

    matrices = []
    matrices.append(matrizA)
    matrices.append((matrizB))
    # Construir matriz de imagenes
    for i in range(len(matrices)):
        matrices[i] = matrices[i].flatten()
    matrices = np.array(matrices)
    # promedio dimension-pixel
    muj = matrices.mean(axis=0)
    # matriz centrada
    X_c = []
    for x_i in matrices:
        X_c.append(x_i - muj)
    # Matriz de covarianza
    C = utilidad.matrizDeCovarianza(np.array(X_c))
    print(C)
    # Exportarla para calcular autovectores
    utilidad.write(np.matrix(C))


