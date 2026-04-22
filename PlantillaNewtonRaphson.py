import math
from Metodos import MetodoNewton_raphson as mnr 

# =============================================================================
# DEFINICIÓN DE LAS FUNCIONES f(x) Y SUS DERIVADAS f'(x)
# =============================================================================

def funcion(x):
    return (x - 1)**2

def funcionDerivada(x):
    return 2 * (x - 1)


# =============================================================================
# PARÁMETROS GENERALES
# =============================================================================

semilla = 0

iteraciones = 10
hacerGrafico = True


# =============================================================================
# EJECUCIÓN DE LOS MÉTODOS
# =============================================================================

if __name__ == "__main__":
    mnr.newton_raphson(funcion, funcionDerivada, semilla, iteraciones, hacerGrafico)