import IO
import matplotlib.pyplot as plt

def graficarAutovalores(filename):
    autovalores = IO.leerMatriz(filename)[0][-10:]
    # plt.yscale("log")
    plt.plot(autovalores)
    plt.show()