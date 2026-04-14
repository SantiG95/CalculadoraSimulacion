import numpy as np
import matplotlib.pyplot as plt

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
        if abs(coef) < 1e-12: continue
        signo = '+' if coef >= 0 else '-'
        valor = abs(coef)
        if potencia == 0: termino = f"{valor:.6g}"
        elif potencia == 1: termino = f"{valor:.6g}x"
        else: termino = f"{valor:.6g}x^{potencia}"
        
        if terminos: terminos.append(f" {signo} {termino}")
        else: terminos.append(termino if signo == '+' else f"-{termino}")
    return ''.join(terminos) if terminos else '0'

def cota_error_lagrange(x, x_puntos, max_derivada):
    """
    Calcula la cota superior del error global usando el máximo de la 
    derivada (n+1) en el intervalo.
    """
    n = len(x_puntos)
    # n es el número de puntos, el grado del polinomio es n-1.
    # El error usa la derivada de orden n.
    factor = np.prod([x - xi for xi in x_puntos])
    return abs((max_derivada / np.math.factorial(n)) * factor)

def lagrange(x_puntos, y_puntos, x_eval=None, f=None, max_derivada=None, desborde=False):
    """
    Función principal para interpolar, evaluar y graficar.
    """
    x_plot = np.linspace(min(x_puntos), max(x_puntos), 400)
    if desborde:
        margen = (max(x_puntos) - min(x_puntos)) * 0.2
        x_plot = np.linspace(min(x_puntos) - margen, max(x_puntos) + margen, 400)

    P = polinomio_lagrange_poly(x_puntos, y_puntos)
    y_plot = P(x_plot)

    if x_eval is None:
        x_eval = float(np.mean(x_puntos))

    p_eval = float(P(x_eval))
    error_local = None
    cota_error = None

    if f is not None:
        error_local = abs(f(x_eval) - p_eval)

    if max_derivada is not None:
        cota_error = cota_error_lagrange(x_eval, x_puntos, max_derivada)

    print('\n--- RESULTADOS INTERPOLACIÓN DE LAGRANGE ---')
    print(f'Polinomio P(x): {polinomio_a_texto(P)}')
    print(f'Evaluación en x = {x_eval:.6f}: P({x_eval:.6f}) = {p_eval:.6f}')

    if error_local is not None:
        print(f'Error local exacto (|f(x)-P(x)|): {error_local:.6e}')
    
    if cota_error is not None:
        print(f'Cota de error teórico (Máx): {cota_error:.6e}')
    else:
        print('Nota: Para el error teórico, ingresá el valor máximo de la derivada (n+1).')

    # Gráfico
    plt.figure(figsize=(10, 6))
    plt.scatter(x_puntos, y_puntos, color='red', zorder=5, label='Nodos (puntos dados)')
    plt.plot(x_plot, y_plot, label='Polinomio interpolante $P(x)$', color='blue', linewidth=2)
    plt.scatter([x_eval], [p_eval], color='green', marker='x', s=100, label=f'Evaluación en {x_eval}')
    
    plt.title('Interpolación de Lagrange', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

    return P

if __name__ == '__main__':
    # Ejemplo con Ejercicio 13a: f(x) = cos(x)
    nodos_x = np.array([0, 0.6, 0.9])
    nodos_y = np.cos(nodos_x)
    punto_a_evaluar = 0.45
    
    # La 3ra derivada de cos(x) es sin(x). El máximo de |sin(x)| en [0, 0.9] es sin(0.9)
    max_der = abs(np.sin(0.9)) 
    
    lagrange(nodos_x, nodos_y, x_eval=punto_a_evaluar, f=np.cos, max_derivada=max_der)