import numpy as np
import pandas as pd
import IO
import matplotlib.pyplot as plt


def imprimirImagenes(imagenes, destino=None, shape=(5,2)):
    x, y = shape
    f, axs = plt.subplots(x, y, figsize=(3, 8))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(imagenes[i], cmap=plt.cm.gray);
        ax.axis('off')
    plt.tight_layout()
    if destino:
        plt.savefig(destino)
    plt.show()


def graficarAutovalores(filename, metodo, cantidad):
    autovalores = IO.leerMatriz("resultados/", filename)[0][:cantidad]
    plt.plot(autovalores)
    if metodo == "2DPCA":
        plt.xticks(range(cantidad))
    if metodo == "PCA":
        x_axis = list(range(0, 30, 5)) + list(range(30, 101, 10))
        plt.xticks(x_axis)
    plt.title('Autovalores ordenados de ' + metodo)
    plt.xlabel('Componente Principal')
    plt.ylabel('Autovalor')
    plt.grid(alpha=0.5)
    plt.savefig('resultados/ejercicio_2/item_b/grafico_autovalores_' + metodo)
    plt.clf()


def graficarAutovaloresCompleto(filename, cantidad):
    autovalores = IO.leerMatriz("resultados/", filename)[0][:cantidad]
    plt.plot(autovalores)
    x_axis = [*range(0, cantidad + 1, 50)]
    plt.xticks(x_axis)
    plt.xticks(rotation=90)
    plt.title('Autovalores ordenados de PCA')
    plt.xlabel('Componente Principal')
    plt.ylabel('Autovalor')
    plt.grid(alpha=0.5)
    plt.yscale("log")
    plt.savefig('resultados/ejercicio_2/item_b/grafico_autovalores_completo_pca')
    plt.clf()


def graficarCorrelacion(matrix, label, filename):
    fig, ax = plt.subplots()
    fig.suptitle("Matrices de Correlación", fontsize=16)
    heatmap = ax.pcolor(matrix, cmap= 'RdBu')
    ax.set_title(label)
    ax.set_aspect("equal")
    fig.colorbar(heatmap, ax=ax)
    plt.tight_layout()
    plt.savefig('resultados/ejercicio_3/item_a/' + filename)
    plt.clf()

def graficarEigenFacesPCA(filename, cantidad):
    V = IO.leerMatriz("resultados/", filename, cantidad)
    eigenFaces = []
    for autovector in V:
        v_i = np.reshape(autovector, (56, 46))
        eigenFaces.append(v_i)
    imprimirImagenes(eigenFaces, 'resultados/ejercicio_2/item_c/eigenfaces_pca')

def graficarEigenFacesTDPCA(Y, U, k):
    # Plotea las k primeras eigenfaces de 2DPCA
    eigenFaces = []
    for i in range(k):
        eigenface = np.outer(Y[i], U[i])
        eigenFaces.append(eigenface)
    imprimirImagenes(eigenFaces, 'resultados/ejercicio_2/item_c/eigenfaces_2dpca')


def graficarErrorCompresion(imagenes, imagenes_procesadas, titulo, metodo1="pca", metodo2="tdpca"):
    errores = {metodo1: {}, metodo2:{}}

    for k in imagenes_procesadas[metodo1].keys():
        errores[metodo1][k] = calcularErrorCompresion(imagenes, imagenes_procesadas[metodo1][k])

    for k in imagenes_procesadas[metodo2].keys():
        errores[metodo2][k] = calcularErrorCompresion(imagenes, imagenes_procesadas[metodo2][k])

    for error in errores.values():
        x = error.keys()
        y = error.values()
        plt.scatter(x, y)

    plt.legend(errores.keys())

    plt.title(titulo)
    plt.xlabel('Valores de k')
    plt.ylabel('Error')
    plt.savefig('resultados/ejercicio_3/item_c/' + titulo)
    plt.clf()

def graficarErrorEnRango(imagenes, imagenes_procesadas, metodo):
    errores = {}
    for k in imagenes_procesadas.keys():
        errores[k] = calcularErrorCompresion(imagenes, imagenes_procesadas[k])

    xticks = list(errores.keys())
    _, ax = plt.subplots()
    plt.scatter(xticks, list(errores.values()))

    ax.set_xticks(xticks)
    plt.xticks(rotation=90)
    plt.title("Error de compresión " + metodo)
    plt.xlabel('Cantidad de componentes')
    plt.ylabel('Error')
    plt.grid(alpha=0.5)
    plt.savefig('resultados/ejercicio_3/item_c/rango_' + metodo)
    plt.clf()

def graficarMetricasSimiliaridad(data, titulo):
    df = pd.DataFrame(data).T

    _, ax = plt.subplots()
    df.plot(kind="bar", ax=ax)

    plt.title(titulo)
    plt.xlabel('Valores de k')
    plt.ylabel('Valor metrica')
    plt.grid(alpha=0.3)
    ax.legend(["Mismo", "Distinto"])
    plt.savefig('resultados/ejercicio_3/item_b/' + titulo)
    plt.clf()

def calcularErrorCompresion(imagenes, imagenes_procesadas):
    error = np.zeros(len(imagenes))
    for i in range(len(imagenes)):
        error[i] = np.linalg.norm(np.subtract(imagenes[i], imagenes_procesadas[i]), ord=2)
    return np.mean(error)


def boxplotTiempos(data, rango, metodo):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.boxplot(data)
    ax.set_title("Tiempo de ejecución de " + metodo)
    ax.set_xlabel('Cantidad de componentes')
    ax.set_ylabel('Tiempo (segundos)')

    x = [*range(len(data) + 1)]
    plt.xticks(x, [0] + rango)
    plt.grid()
    plt.savefig("resultados/ejercicio_4/tiempos_" + metodo)
    plt.clf()
