import numpy as np
import seaborn as sns
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
    plt.show()


def graficarCorrelacion(matrix, label, filename):
    fig, ax = plt.subplots()
    fig.suptitle("Matrices de Correlación", fontsize=16)
    heatmap = ax.pcolor(matrix, cmap= 'RdBu')
    ax.set_title(label)
    ax.set_aspect("equal")
    fig.colorbar(heatmap, ax=ax)
    plt.tight_layout()
    plt.savefig('resultados/ejercicio_3/item_a/' + filename)
    #plt.show()
    plt.clf()

def graficarEigenFacesPCA(filename, cantidad):
    V = IO.leerMatriz("resultados/", filename, cantidad)
    eigenFaces = []
    for autovector in V:
        v_i = np.reshape(autovector, (56, 46))
        eigenFaces.append(v_i)
    imprimirImagenes(eigenFaces, 'resultados/ejercicio_2/item_c/eigenfaces_pca')

# def graficarEigenFacesTDPCA(Z):

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
    plt.show()

def calcularErrorCompresion(imagenes, imagenes_pca):
    error = np.zeros(len(imagenes))
    for i in range(len(imagenes)):
        error[i] = np.linalg.norm(np.subtract(imagenes[i], imagenes_pca[i]), ord=2)
    return np.mean(error)
