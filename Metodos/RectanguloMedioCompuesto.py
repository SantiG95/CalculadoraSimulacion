import numpy as np

def rectanguloMedioCompuesto(funcion, inicio, fin, n):
    h = (fin - inicio) / n
    x_medio = np.linspace(inicio + h/2, fin - h/2, n)

    integral = h * np.sum(funcion(x_medio))
    return integral