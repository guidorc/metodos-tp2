import numpy as np
import utils
import ejecutar
import IO
import plotter
import config
import time

def obtenerMatricesCovarianzayCorrelación(X, sufijo = ''):
    # centrar matriz
    X_c = utils.centrarMatriz(X)
    # Matriz de covarianza
    print("Calculando Matriz de Covarianza")
    C = utils.matrizDeCovarianza(X_c)
    print("Calculando Matriz de Correlación")
    R = utils.matrizDeCorrelación(X_c)
    # Exportarla para calcular autovalores y autovectores
    print("Escribiendo Matriz de Covarianza")
    IO.write(np.matrix(C), "covarianza" + sufijo + ".txt")
    print("Escribiendo Matriz de Correlación")
    IO.write(np.matrix(R), "correlacion" + sufijo + ".txt")
    return C

def proyectarPCA(V, X, k):
    Z = []
    for i in range(len(X)):
        z_i = utils.proyectar(V, X[i], k)
        Z.append(z_i)
    return np.array(Z)

def reconstruirPCA(V, Z, k, h, w):
    imagenes_reconstruidas = []
    for i in range(len(Z)):
        imagenes_reconstruidas.append(utils.reconstruirImagen(Z[i], V, k))
    imagenes_formateadas = utils.formatearImagenes(imagenes_reconstruidas, h, w)
    return imagenes_formateadas


def PCA(imagenes, k, calcularCovarianza = False, autovectores="covarianza_eigenVectors.csv", omitirAutovectores=True):
    # -------- PCA -------- #
    X = utils.aplanarImagenes(imagenes)
    # Obtener componentes principales
    if calcularCovarianza:
        obtenerMatricesCovarianzayCorrelación(X)
        # Calcular autovalores y autovectores de matriz de covarianza
        print("Ejecutando Deflacion para Matriz de Covarianza")
        if omitirAutovectores:
            ejecutar.correrTp("covarianza", k)
        else:
            ejecutar.correrTp("covarianza")
    V = np.array(IO.leerMatriz("resultados/", autovectores, k))
    # Obtener proyeccion de menor dimension
    Z = proyectarPCA(V, X, k)
    # Reconstruir imagenes
    _, h, w = imagenes.shape
    return reconstruirPCA(V, Z, k, h, w), Z


def reconstruirTDPCA(M, U, k):
    h = len(M[0][0])
    w = len(U[0])
    # U_t = np.transpose(U)
    imagenes_reconstruidas = []
    for V in M:
        A = np.zeros((h, w))
        for i in range(k):
            Y_i = V[i]
            X_i = U[i]
            A += np.outer(Y_i, X_i)
        imagenes_reconstruidas.append(A)
    for i, imagen in enumerate(imagenes_reconstruidas):
        IO.write(imagen, str(i) + '.pgm', 'resultados/caras/s1')
    return np.array(imagenes_reconstruidas)

def TDPCA(imagenes, k, calcularAutovectores=False, autovectores="covarianza_2dpca_eigenVectors.csv", eigenfaces=False):
    # Calculo image covariance matrix
    G = utils.imageCovarianceMatrix(imagenes)
    # Calculo matriz de correlacion
    R = utils.matrizDeCorrelación(G)
    # Calculo base de autovectores
    if calcularAutovectores:
        # Calcular autovectores de G
        IO.write(np.matrix(G), "covarianza_2dpca.txt")
        IO.write(np.matrix(R), "correlacion_2dpca.txt")
        ejecutar.correrTp("covarianza_2dpca", k)
    U = IO.leerMatriz("resultados/", autovectores, k)
    # Calcular feature vectors
    feature_matrix = []  # de n x a x b
    for A in imagenes:
        # Calculo matriz de feature vectors para A de a x b
        Y = []  # Y de a x b
        for i in range(len(U)):
            X_i = U[i]
            Y.append(np.matmul(A, X_i))
        feature_matrix.append(Y)
    # Obtener proyeccion de menor dimension
    Z = []
    for imagen in feature_matrix:
        z_i = imagen[:k]
        Z.append(z_i)
    # Para obtener las eigenfaces usamos los feature vectors y los autovectores
    if eigenfaces:
        return feature_matrix, U
    # Retorna imagenes reconstruidas y su proyección
    return reconstruirTDPCA(feature_matrix, U, k), np.array(Z)


def graficarAutovalores():
    # PCA
    plotter.graficarAutovalores("covarianza_eigenValues.csv", "PCA", 100)
    plotter.graficarAutovaloresCompleto("covarianza_eigenValues.csv", 900)
    # 2DPCA
    plotter.graficarAutovalores("covarianza_2dpca_eigenValues.csv", "2DPCA", 20)

def generarEigenFaces(imagenes, k):
    # PCA:
    plotter.graficarEigenFacesPCA("covarianza_eigenVectors.csv", 10)
    # 2DPCA
    features, autovectores = TDPCA(imagenes, k, False, "covarianza_2dpca_eigenVectors.csv", True)
    plotter.graficarEigenFacesTDPCA(features[0], autovectores, 10)


def regenerarRostros(imagenes):
    folder = "resultados/ejercicio_2/item_d/"
    # PCA
    for componentes in [10, 100, 200, 300, 400]:
        rostros_pca, _ = PCA(imagenes, componentes, False)
        filename = 'rostros_pca_' + str(componentes)
        plotter.imprimirImagenes(rostros_pca, folder + filename)
    # 2DPCA
    for componentes in [5, 10, 20, 30, 40]:
        rostros_tdpca, _ = TDPCA(imagenes, componentes, False)
        filename = 'rostros_2dpca_' + str(componentes)
        plotter.imprimirImagenes(rostros_tdpca, folder + filename)


def generarCorrelacion(imagenes, rango_pca, rango_tdpca):
    # Conjunto original
    X = utils.aplanarImagenes(imagenes)
    obtenerMatricesCovarianzayCorrelación(X)
    # PCA
    for componentes in rango_pca:
        _, z_pca = PCA(imagenes, componentes, False)
        obtenerMatricesCovarianzayCorrelación(z_pca, "_pca_" + str(componentes))
    # 2DPCA
    for componentes in rango_tdpca:
        _, z_tdpca = TDPCA(imagenes, componentes, False)
        z_aplanada = utils.aplanarImagenes(z_tdpca)
        obtenerMatricesCovarianzayCorrelación(z_aplanada, "_2dpca_" + str(componentes))

def calcularMetricas(M):
    r, c = np.shape(M)
    contador = 0
    suma_mismo = 0
    suma_distinto = 0
    divisor_mismo = 41 * 100
    divisor_distinto = (r * c) - divisor_mismo

    for i in range(r-1, -1, -1):
        offset = c - contador * 10
        for j in range(c-1, -1, -1):
            coef = M[i][j]
            if offset > j >= (offset - 10):
                suma_mismo += coef
            else:
                suma_distinto += coef
        if i % 10 == 0:
            contador += 1

    return (suma_mismo / divisor_mismo), (suma_distinto / divisor_distinto)


def errorPcaVsTdpca(imagenes):
    ks_pca = list(range(50, 401, 50))
    ks_tdpca = list(range(5, 46, 5))
    imagenes_procesadas = {"pca": {}, "tdpca": {}}
    for k in ks_pca:
        imagenes_pca, z_pca = PCA(imagenes, k, False)
        imagenes_procesadas["pca"][k] = imagenes_pca
    for k in ks_tdpca:
        imagenes_tdpca, z_tdpca = TDPCA(imagenes, k, False)
        imagenes_procesadas["tdpca"][k] = imagenes_tdpca
    plotter.graficarErrorCompresion(imagenes, imagenes_procesadas, "Error de compresión PCA vs 2DPCA")

def errorEnRango(imagenes, metodo, rango, calcularCovarianza=False):
    imagenes_procesadas = {}
    if metodo == "pca":
        for k in rango:
            print("resolviendo para k: " + str(k))
            imagenes_pca, z_pca = PCA(imagenes, k, calcularCovarianza)
            imagenes_procesadas[k] = imagenes_pca
    else:
        for k in rango:
            print("resolviendo para k: " + str(k))
            imagenes_tdpca, z_tdpca = TDPCA(imagenes, k, calcularCovarianza)
            imagenes_procesadas[k] = imagenes_tdpca
    plotter.graficarErrorEnRango(imagenes, imagenes_procesadas, metodo)


def visualizarCorrelacion(imagenes, generarMatrices=False):
    if generarMatrices:
        generarCorrelacion(imagenes, [50, 400], [5, 40])

    filenames = ["correlacion", "correlacion_pca_50", "correlacion_pca_400", "correlacion_2dpca_5",
                 "correlacion_2dpca_40"]
    labels = ["Conjunto de datos original", "Datos procesados con PCA para k=50", "Datos procesados con PCA para k=400",
              "Datos procesados con 2DPCA para k=5", "Datos procesados con 2DPCA para k=40"]

    data = IO.leerMatricesCorrelacion(filenames)
    for matriz, label, filename in zip(data, labels, filenames):
        plotter.graficarCorrelacion(matriz, label, filename)


def compararMetricas(imagenes, metodo, rango, generarMatrices=False):
    if generarMatrices:
        if (metodo == "pca"):
            generarCorrelacion(imagenes, rango, [])
        else:
            generarCorrelacion(imagenes, [], rango)
    filenames = ["correlacion_" + metodo + "_" + str(x) for x in rango]
    matrices = IO.leerMatricesCorrelacion(filenames)
    data = {}
    for m, k in zip(matrices, rango):
        mismo, distinto = calcularMetricas(m)
        data[k] = [mismo, abs(distinto)]
    plotter.graficarMetricasSimiliaridad(data, "Metricas " + metodo)

def errorSetReducido(imagenes, metodo, ks):
    imagenes_procesadas = {"completo": {}, "reducido": {}}
    for k in ks:
        imagenes_completo, _ = metodo(imagenes, k, False)
        imagenes_reducido, _ = metodo(imagenes, k, False, metodo.__name__ + "_autovectores_menos_una.csv")
        imagenes_procesadas["completo"][k] = imagenes_completo
        imagenes_procesadas["reducido"][k] = imagenes_reducido
    plotter.graficarErrorCompresion(imagenes, imagenes_procesadas, metodo.__name__ + ": Error de compresión set reducido", "completo", "reducido")


def medirTiempos(metodo, rango):
    for k in rango:
        tiempos = []
        for iter in range(0, 20):
            if metodo == "pca":
                start = time.time()
                PCA(imagenes, k, True)
                end = time.time()
                tiempos.append(end-start)
            else:
                start = time.time()
                TDPCA(imagenes, k, True)
                end = time.time()
                tiempos.append(end - start)

        IO.writeTimes(tiempos, metodo + "_" + str(k))


def tiempoSegunComponentes(metodo, rango, ejecutar=False):
    if ejecutar:
        medirTiempos(metodo, rango)

    tiempos = {}
    for k in rango:
        filename = metodo + "_" + str(k)
        tiempos[k] = np.loadtxt("resultados/ejercicio_4/" + filename, delimiter=" ")

    data = [v for v in tiempos.values()]
    plotter.boxplotTiempos(data, rango, metodo)


if __name__ == '__main__':
    # Leer caras
    imagenes = IO.cargarImagenes()
    k_pca = config.k_pca
    k_2dpca = config.k_2dpca

    # Ejecutar PCA y 2DPCA para luego utilizar resultados
    #PCA(imagenes, 901, True)
    #TDPCA(imagenes, k_2dpca, True)

    # -------- EXPERIMENTACION -------- #
    # Ejercicio 2
    # b) Observar autovalores de mayor a menor
    graficarAutovalores()

    # c) Observar eigenfaces
    #generarEigenFaces(imagenes, k_2dpca)

    # d) Regeneramos rostros de la primer persona, para distintos valores de k
    #regenerarRostros(imagenes)

    # Ejercicio 3
    # a) Visualizar matriz de correlación
    #visualizarCorrelacion(imagenes, True)

    # b) Metricas de similaridad
    #compararMetricas(imagenes, "pca", [*range(50, 401, 50)], generarMatrices=False)
    #compararMetricas(imagenes, "2dpca", [*range(5, 41, 5)], generarMatrices=False)

    # c) Error de compresion
    #errorPcaVsTdpca(imagenes)
    #errorSetReducido(imagenes, PCA, list(range(50, 401, 50)))
    #errorSetReducido(imagenes, TDPCA, list(range(5, 46, 5)))
    #errorEnRango(imagenes, "pca", [*range(50, 901, 50)])

    # Ejercicio 4
    # Medir tiempos
    #tiempoSegunComponentes("pca", [*range(10, 41, 10)], ejecutar=False)
    #tiempoSegunComponentes("2dpca", [*range(10, 41, 10)], ejecutar=False)
