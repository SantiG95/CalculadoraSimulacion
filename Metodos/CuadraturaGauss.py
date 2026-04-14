import numpy as np
from tabulate import tabulate

'''Calcula la integral definida usando Cuadratura de Gauss-Legendre. 
f: función a integrar. 
a, b: límites de integración. 
n: número de puntos (nodos).'''

def cuadratura_gauss(f, a, b, n):
    # 1. Obtener nodos (xi) y pesos (wi) del intervalo estándar [-1, 1]
    nodos_estandar, pesos = np.polynomial.legendre.leggauss(n)
    filas_tabla = []
    suma_total = 0
    # 2. Iterar para transformar los puntos y evaluar
    for i in range(n):
        # Transformación lineal al intervalo [a, b]
        xn = 0.5 * ((b - a) * nodos_estandar[i] + (b + a))
        # Evaluación de la función en el nodo transformado
        f_xn = f(xn)
        # El aporte a la sumatoria (wi * f(xi))
        # Nota: El peso wi no cambia con la transformación, 
        # pero f(xi) se evalúa en el xn transformado.
        aporte = pesos[i] * f_xn
        suma_total += aporte
        # Guardar datos con 10 decimales para la tabla
        filas_tabla.append([
            i + 1, 
            f"{xn:.10f}", 
            f"{f_xn:.10f}", 
            f"{pesos[i]:.10f}", # Agregamos la columna de pesos para que sea más completo
            f"{aporte:.10f}"
        ])
    # 3. Cálculo final aplicando el factor (b-a)/2
    resultado = 0.5 * (b - a) * suma_total
    # Imprimir la tabla detallada (como en tus otros métodos)
    headers = ["n", "Xn (Nodo)", "f(Xn)", "Peso (wi)", "wi * f(Xn)"]
    print(f"\n--- Cuadratura de Gauss (n={n}) en [{a}, {b}] ---")
    print(tabulate(filas_tabla, headers=headers, tablefmt="fancy_grid", disable_numparse=True))
    print(f"RESULTADO DE LA INTEGRAL: {resultado:.12f}\n")
    return resultado