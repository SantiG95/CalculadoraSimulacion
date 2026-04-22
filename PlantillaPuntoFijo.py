import math
from Metodos import MetodoPuntoFijo as mpf 

# =============================================================================
# DEFINICIÓN DE LAS FUNCIONES g(x)
# Recordatorio: Despejamos 'x' de la ecuación original f(x) = 0 para obtener x = g(x)
# =============================================================================

def g1(x):
    return (2 * math.exp(x**2)) / 5


limiteInferior = 0 
limiteSuperior = 1 

x0 = 0

iteraciones = 20

hacerGrafico = True


# =============================================================================
# EJECUCIÓN DE LOS MÉTODOS
# =============================================================================

mpf.punto_fijo_con_rango(g1, limiteInferior, limiteSuperior, x0, iteraciones, tolerancia=1e-4)

mpf.punto_fijo(g1, x0, iteraciones, hacerGrafico, tolerancia=1e-4)
