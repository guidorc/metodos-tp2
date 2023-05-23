import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def cargarImagenes():
    paths = []
    imgs = []
    for path in sorted(list(Path('matrices/caras').rglob('*/*.pgm'))):
        paths.append(path)
        image = (plt.imread(path)/255)
        imgs.append(image)
    result = np.stack(imgs)
    return result
