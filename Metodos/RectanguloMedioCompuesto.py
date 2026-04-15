import numpy as np

def rectanguloMedioCompuesto(funcion, inicio, fin, n):
    h = (fin - inicio) / n
    x_medio = np.linspace(inicio + h/2, fin - h/2, n)

    integral = h * np.sum(funcion(x_medio))
    return integral

def rectanguloIzquierdoCompuesto(funcion, inicio, fin, n):
    h = (fin - inicio) / n
    # Los x van desde 'inicio' hasta el penúltimo punto (no toca 'fin')
    x_izq = np.linspace(inicio, fin - h, n)
    integral = h * np.sum(funcion(x_izq))
    return integral

def rectanguloDerechoCompuesto(funcion, inicio, fin, n):
    h = (fin - inicio) / n
    # Los x van desde el segundo punto hasta 'fin' (no toca 'inicio')
    x_der = np.linspace(inicio + h, fin, n)
    integral = h * np.sum(funcion(x_der))
    return integral