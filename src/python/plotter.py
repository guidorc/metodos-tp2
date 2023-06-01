import numpy as np
import seaborn as sns
import IO
import matplotlib.pyplot as plt


def imprimirImagenes(imagenes):
    f, axs = plt.subplots(5, 2, figsize=(3, 8))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(imagenes[i], cmap=plt.cm.gray);
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def graficarAutovalores(filename):
    autovalores = IO.leerMatriz("resultados/", filename)[0][-10:]
    # plt.yscale("log")
    plt.plot(autovalores)
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

def graficarMetricasSimiliaridad(data, labels):
    # :)
    print("Esta función está en construcción, disculpe las molestias ocasionadas. -Metrovías")