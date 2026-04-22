import numpy as np
from scipy.integrate import quad # ¡Clave para calcular la integral exacta!

# Importamos tus métodos
from Metodos import TrapecioCompuesta as mt
from Metodos import Simpson1_3 as s13
from Metodos import Simpson3_8 as s38
from Metodos import ErrorRelativoPorcentual as erp 

# =============================================================================
# DEFINICIÓN DE FUNCIONES A INTEGRAR
# =============================================================================

# Ejercicio 1: f(x) = 6 + 3*cos(x)
def funcion(x):
    return 6 + 3 * np.cos(x)

inicio = 0
fin = np.pi / 2

# Variables independientes para cada método (según lo que pida la guía)
n = 6

# =============================================================================
# EJECUCIÓN DE LOS MÉTODOS
# =============================================================================

if __name__ == "__main__":
    
    integral_exacta, error_estimado = quad(funcion, inicio, fin)
    print(f"=== SOLUCIÓN ANALÍTICA (EXACTA) ===")
    print(f"Valor: {integral_exacta:.12f}\n")
    
    # -------------------------------------------------------------------------
    print("=== a) Trapecio ===")
    aprox_trapecio = mt.trapecioCompuesta(funcion, inicio, fin, n)
    
    err_trap = abs((integral_exacta - aprox_trapecio) / integral_exacta) * 100
    print(f"Error Relativo Porcentual: {err_trap:.6f}%\n")
    
    # -------------------------------------------------------------------------
    print("=== b) Simpson 1/3 ===")
    aprox_simpson13 = s13.simpson1_3(funcion, inicio, fin, n)
    
    err_s13 = abs((integral_exacta - aprox_simpson13) / integral_exacta) * 100
    print(f"Error Relativo Porcentual: {err_s13:.6f}%\n")
    
    # -------------------------------------------------------------------------
    print("=== c) Simpson 3/8 ===")
    aprox_simpson38 = s38.simpson3_8(funcion, inicio, fin, n)
    
    err_s38 = abs((integral_exacta - aprox_simpson38) / integral_exacta) * 100
    print(f"Error Relativo Porcentual: {err_s38:.6f}%\n")