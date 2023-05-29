import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def cargarImagenes():
    paths = []
    imgs = []
    for path in sorted(list(Path('matrices/caras/s1').rglob('*.pgm'))):
        paths.append(path)
        image = (plt.imread(path)[::2, ::2]/255)
        imgs.append(image)
    result = np.stack(imgs)
    return result

def leerMatriz(filename):
    matrix = open("resultados/" + filename)
    return np.genfromtxt(matrix, delimiter=",", names=True, dtype=None, unpack=True)

def imprimirImagenes(imagenes):

    # Esto imprime una sola imagen pero funciona
    print(imagenes[0].shape)
    plt.imshow(imagenes[0], cmap="gray")
    plt.show()

    # Este es el código en el que me estoy basando para imprimir todas las imágenes, tiene un problema de tamaños que voy a arreglar más tarde
    h = imagenes.shape[1]
    w = imagenes.shape[2]
    f, axs = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(imagenes[i], cmap=plt.cm.gray);
        ax.axis('off')
    plt.tight_layout()
