import math
import numpy as np
from Metodos import MetodoAitken as ma

# =============================================================================
# DEFINICIÓN DE LAS FUNCIONES g(x)
# =============================================================================

def funcion(x):
    return (math.pi / 2) * x**2 - 2

# =============================================================================
# PARÁMETROS GENERALES
# =============================================================================

x0 = 1.4
iteracionesMaximas = 20
tolerancia = 1e-4

hacerGrafico = True

# =============================================================================
# EJECUCIÓN DE LOS MÉTODOS
# =============================================================================

if __name__ == "__main__":
    
    ma.aceleracion_aitken(funcion, x0, iteraciones=iteracionesMaximas, tol=tolerancia, hacerGrafico=hacerGrafico)
