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
    autovalores = IO.leerMatriz(filename)[0][-10:]
    # plt.yscale("log")
    plt.plot(autovalores)
    plt.show()
