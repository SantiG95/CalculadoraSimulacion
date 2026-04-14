import numpy as np
from tabulate import tabulate
# Importamos la cuarta derivada desde tu paquete
from .Derivada import cuarta_derivada 
from .ErrorRelativoPorcentual import error_relativo_porcentual

def simpson3_8(funcion, inicio, fin, n):
    if (n % 3) != 0:
        raise ValueError("Para Simpson 3/8, 'n' debe ser múltiplo de 3 (3, 6, 9...).")

    h = (fin - inicio) / n
    x = np.linspace(inicio, fin, n + 1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        y = funcion(x)
   
    if np.isnan(y).any():
        y = np.where(np.isnan(y), 1.0, y)

    # Cálculo de la Integral
    suma_extremos = y[0] + y[-1]
    indices = np.arange(1, n)
    mask_multiplo_3 = (indices % 3 == 0)
    
    suma_pesan_3 = 3 * np.sum(y[indices[~mask_multiplo_3]])
    suma_pesan_2 = 2 * np.sum(y[indices[mask_multiplo_3]])
    
    s = suma_extremos + suma_pesan_3 + suma_pesan_2
    integral = (3 * h / 8) * s

    # --- CÁLCULO DEL ERROR USANDO TU ARCHIVO DERIVADA ---
    punto_medio = (inicio + fin) / 2
    f4_val = cuarta_derivada(funcion, punto_medio)
    
    # Usamos la fórmula de la imagen 2 adaptada a la cantidad de segmentos
    # Dividimos por (n/3) porque el error se acumula por cada bloque de 3
    error_truncamiento = - (((fin - inicio)**5) / (6480 * (n/3)**4)) * f4_val
    # ----------------------------------------------------

    filas_tabla = []
    for i in range(len(x)):
        if i == 0 or i == n:
            coef = 1
        elif i % 3 == 0:
            coef = 2
        else:
            coef = 3
        filas_tabla.append([i, f"{x[i]:.10f}", f"{y[i]:.10f}", coef])

    print("\n--- Detalle de evaluación (Simpson 3/8) ---")
    print(tabulate(filas_tabla, headers=["i", "Xi", "f(Xi)", "Coef."], tablefmt="fancy_grid"))
    
    print(f"\nRESULTADO INTEGRAL: {integral:.12f}")
    print(f"ERROR DE TRUNCAMIENTO: {error_truncamiento:.2e}")

    return integral