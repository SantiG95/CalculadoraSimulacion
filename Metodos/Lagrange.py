import numpy as np
import matplotlib.pyplot as plt

def polinomio_lagrange(x, x_puntos, y_puntos):
    n = len(x_puntos)
    L = 0
    for i in range(n):
        li = 1
        for j in range(n):
            if i != j:
                li *= (x - x_puntos[j])/(x_puntos[i]-x_puntos[j])
        L += y_puntos[i] * li
    return L


def polinomio_lagrange_poly(x_puntos, y_puntos):
    n = len(x_puntos)
    P = np.poly1d([0])
    for i in range(n):
        Li = np.poly1d([1])
        for j in range(n):
            if i != j:
                Li *= np.poly1d([1, -x_puntos[j]]) / (x_puntos[i] - x_puntos[j])
        P += y_puntos[i] * Li
    return P


def polinomio_a_texto(poly):
    coeficientes = poly.coeffs
    grado = len(coeficientes) - 1
    terminos = []
    for i, coef in enumerate(coeficientes):
        potencia = grado - i
        if abs(coef) < 1e-12:
            continue
        signo = '+' if coef >= 0 else '-'
        valor = abs(coef)
        if potencia == 0:
            termino = f"{valor:.6g}"
        elif potencia == 1:
            termino = f"{valor:.6g}x"
        else:
            termino = f"{valor:.6g}x^{potencia}"
        if terminos:
            terminos.append(f" {signo} {termino}")
        else:
            terminos.append(termino if signo == '+' else f"-{termino}")
    return ''.join(terminos) if terminos else '0'


def error_teorico_lagrange(x, x_puntos, f_nth_derivative):
    n = len(x_puntos)
    factor = np.prod([x - xi for xi in x_puntos])
    return abs(f_nth_derivative(x) / np.math.factorial(n) * factor)

    

def lagrange(x_puntos, y_puntos, x_eval=None, f=None, f_nth_derivative=None, desborde=False):
    x = np.linspace(min(x_puntos), max(x_puntos), 400)
    if desborde:
        x = np.linspace(min(x_puntos) - 1, max(x_puntos) + 1, 400)

    P = polinomio_lagrange_poly(x_puntos, y_puntos)
    y = P(x)

    if x_eval is None:
        x_eval = float(np.mean(x_puntos))

    p_eval = float(P(x_eval))
    error_local = None
    error_teorico = None

    if f is not None:
        error_local = abs(f(x_eval) - p_eval)

    if f_nth_derivative is not None:
        error_teorico = error_teorico_lagrange(x_eval, x_puntos, f_nth_derivative)

    print('\nPolinomio interpolante de Lagrange:')
    print(polinomio_a_texto(P))
    print(f'P({x_eval:.6f}) = {p_eval:.6f}')

    if error_local is not None:
        print(f'Error local en x={x_eval:.6f}: {error_local:.6e}')
    else:
        print('Error local: no se puede calcular sin la función real f(x).')

    if error_teorico is not None:
        print(f'Error teórico en x={x_eval:.6f}: {error_teorico:.6e}')
    else:
        print('Error teórico: no se puede calcular sin la derivada n-ésima de f(x).')

    plt.scatter(x_puntos, y_puntos, color='red', label='Puntos dados')
    plt.plot(x, y, label='Polinomio de Lagrange')
    plt.legend()
    plt.grid(True)
    plt.show()

    return P, error_local, error_teorico


if __name__ == '__main__':
    x_puntos = np.array([0, 1, 2])
    y_puntos = np.array([1, 3, 0])
    lagrange(x_puntos, y_puntos, desborde=True)







