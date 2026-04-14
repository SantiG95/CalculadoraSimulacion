import numpy as np
from tabulate import tabulate
# Importamos la cuarta derivada desde tu propio módulo de derivadas
from .Derivada import cuarta_derivada 

def simpson1_3(funcion, inicio, fin, n):
    if (n % 2) != 0:
        raise ValueError("Para Simpson 1/3, 'n' debe ser par.")

    h = (fin - inicio) / n
    x = np.linspace(inicio, fin, n + 1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        y = funcion(x)
    
    if np.isnan(y).any():
        y = np.where(np.isnan(y), 1.0, y)

    integral = (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n:2]))

    # --- CÁLCULO DEL ERROR LLAMANDO A TU MÓDULO ---
    punto_medio = (inicio + fin) / 2
    # Llamamos a tu función de Derivada.py
    f4_val = cuarta_derivada(funcion, punto_medio)
    
    error_truncamiento = - ((fin - inicio)**5 / (180 * (n**4))) * f4_val
    # ----------------------------------------------

    filas_tabla = []
    for i in range(len(x)):
        coef = 1 if (i == 0 or i == n) else (4 if i % 2 != 0 else 2)
        filas_tabla.append([i, f"{x[i]:.5f}", f"{y[i]:.10f}", coef])

    print("\n--- Detalle de evaluación (Simpson 1/3) ---")
    print(tabulate(filas_tabla, headers=["i", "Xi", "f(Xi)", "Coef."], tablefmt="fancy_grid"))
    
    print(f"\nRESULTADO INTEGRAL: {integral:.12f}")
    print(f"ERROR DE TRUNCAMIENTO: {error_truncamiento:.2e}")
    
    return integral