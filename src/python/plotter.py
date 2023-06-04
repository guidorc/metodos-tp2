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

def leerMatrices():
    filenames = ["correlacion.txt", "correlacion_pca_100.txt", "correlacion_pca_100.txt", "correlacion_tdpca_10.txt", "correlacion_tdpca_10.txt"]
    labels = ["Conjunto de datos original", "Datos procesados con PCA para k=100", "Datos procesados con PCA para k=100", "Datos procesados con 2DPCA para k=10", "Datos procesados con 2DPCA para k=10"]
    data = []
    for i, filename in enumerate(filenames):
        # print("Leyendo ", filename)
        data.append(IO.leerMatrizCorrelacion("matrices/", filename))
    return data, labels
def graficarCorrelacion(data, labels):
    fig, axs = plt.subplots(1, 5, figsize=(18, 4))

    fig.suptitle("Matrices de Correlación", fontsize=16)

    for i, ax in enumerate(axs):
        heatmap = ax.pcolor(data[i], cmap= 'GnBu')
        # print(data[i][:10])

        # Labels
        ax.set_title(labels[i])
        # ax.set_xlabel("X-axis")
        # ax.set_ylabel("Y-axis")

        # Colorbar
        cbar = fig.colorbar(heatmap, ax=ax)
        # cbar.set_label('Colorbar Label')

    # plt.tight_layout()
    plt.show()

def graficarEigenFacesPCA(filename, cantidad):
    V = IO.leerMatriz("resultados/", filename, cantidad)
    eigenFaces = []
    for autovector in V:
        v_i = np.reshape(autovector, (56, 46))
        eigenFaces.append(v_i)
    imprimirImagenes(eigenFaces, 'resultados/ejercicio_2/item_c/eigenfaces_pca')

# def graficarEigenFacesTDPCA(Z):

def graficarErrorCompresion(imagenes, imagenes_procesadas):
    errores = {"PCA": {}, "2DPCA":{}}

    for k in imagenes_procesadas["pca"].keys():
        errores["PCA"][k] = calcularErrorCompresion(imagenes, imagenes_procesadas["pca"][k])

    # errores_tdpca = {}
    for k in imagenes_procesadas["tdpca"].keys():
        errores["2DPCA"][k] = calcularErrorCompresion(imagenes, imagenes_procesadas["tdpca"][k])

    for error in errores.values():
        x = error.keys()
        y = error.values()
        plt.scatter(x, y)

    plt.legend(errores.keys())

    plt.title('Error de Compresión')
    plt.xlabel('Valores de k')
    plt.ylabel('Error')
    plt.show()

def calcularErrorCompresion(imagenes, imagenes_pca):
    error = np.zeros(len(imagenes))
    for i in range(len(imagenes)):
        error[i] = np.linalg.norm(np.subtract(imagenes[i], imagenes_pca[i]), ord=2)
    return np.mean(error)
