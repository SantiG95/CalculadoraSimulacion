import numpy as np
from tabulate import tabulate
# Importamos la segunda derivada desde tu paquete
from .Derivada import segunda_derivada_central
from .ErrorRelativoPorcentual import error_relativo_porcentual

def trapecioCompuesta(funcion, inicio, fin, n):
    h = (fin - inicio) / n
    x = np.linspace(inicio, fin, n + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = funcion(x)
    if np.isnan(y).any():
        y = np.where(np.isnan(y), 1.0, y)
        
    # Cálculo de la Integral
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:n]) + y[-1])
    # --- CÁLCULO DEL ERROR SEGÚN TU IMAGEN ---
    punto_medio = (inicio + fin) / 2
    h_derivada = 1e-5  # Paso pequeño para calcular la derivada
    # Llamamos a tu función para la 2da derivada
    f2_val = segunda_derivada_central(funcion, punto_medio, h_derivada)
    # Aplicamos la fórmula: E = - [ (b-a)^3 / (12 * n^2) ] * f''(xi)
    error_truncamiento = - ((fin - inicio)**3 / (12 * (n**2))) * f2_val
    # ------------------------------------------
    datos_puntos = [[i, f"{x[i]:.5f}", f"{y[i]:.10f}"] for i in range(len(x))]
    print("\n--- Detalle de evaluación (Trapecio Compuesto) ---")
    print(tabulate(datos_puntos, headers=["i", "Xi", "f(Xi)"], tablefmt="psql", disable_numparse=True))
    print(f"\nRESULTADO INTEGRAL: {integral:.12f}")
    print(f"ERROR DE TRUNCAMIENTO EST.: {error_truncamiento:.2e}")
    
    return integral