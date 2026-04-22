import numpy as np
from Metodos import Lagrange as ml # Asumo que mantenés la misma estructura de carpetas


# =============================================================================
# PARÁMETROS GENERALES
# =============================================================================

# Puntos dados
PuntosX = np.array([1, 2, 3])
PuntosY = np.array([1, 4, 9])

# Punto a evaluar
XaEvaluar = 4

# Cuando dan la función
def funcion(x):
    return 2 * np.sin(np.pi * x / 6)

# Para cuando se pide evaluar la función en un punto específico (Ejercicio 13b)
derivadaMaxima = 0.5

# Para cuando se pide evaluar un punto por fuera del intervalo de nodos X (Extrapolación)
desborde = True

# =============================================================================
# EJECUCIÓN DE LOS MÉTODOS
# =============================================================================

if __name__ == "__main__":
    
    ml.lagrange(PuntosX, PuntosY)

    #ml.lagrange(PuntosX, PuntosY, XaEvaluar, funcion, desborde)

    #ml.lagrange(PuntosX, PuntosY, XaEvaluar, funcion, desborde, derivadaMaxima)