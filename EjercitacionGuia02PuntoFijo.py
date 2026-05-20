from Metodos.MetodoPuntoFijo import punto_fijo, punto_fijo_con_rango
import numpy as np

# Ejercicio 1
def g1(x):
    return (2*np.e**(x**2))/5

#punto_fijo_con_rango(g1, 0, 1, 0)

#------------------------------------------------------------------
# Ejercicio 2
def g2(x):
    return np.cos(x)

#punto_fijo_con_rango(g2, 1, 2, 1)

# Ejercicio 3
def g3(x):
    return np.exp(-x)

#punto_fijo_con_rango(g3, 0, 1, 0)
#punto_fijo(g3, 0, 20, False)

#------------------------------------------------------------------
# Ejercicio 4
def g4(x):
    return (x + 1)**(1/3)

#punto_fijo_con_rango(g4, 1, 2, 1)

#------------------------------------------------------------------
# Ejercicio 5
def g5(x):
    return np.pi + 0.5 * np.sin(x / 2)

#punto_fijo_con_rango(g5, 0, 2*np.pi, 0)

#------------------------------------------------------------------
# Ejercicio 6
def g6_a(x):
    return abs(3 + x - 2*x**2)**0.25 

def g6_b(x):
    return abs((x + 3 - x**4) / 2)**0.5

punto_fijo(g6_a, x0=1.0, iteraciones=20, hacerGrafico=False)
punto_fijo(g6_b, x0=1.0, iteraciones=20, hacerGrafico=False)

#------------------------------------------------------------------
# Ejercicio 7
def g7(x):
    return 2**(-x)

#punto_fijo_con_rango(g7, rangoInferior=1/3, rangoSuperior=1.0, x0=0.5, iteraciones=20)

#------------------------------------------------------------------
# Ejercicio 8
def g8(x):
    return (x + 3) / (x + 1)

punto_fijo(g8, 0, 20) 

#------------------------------------------------------------------
# Ejercicio 9
def g9(x):
    return 5 / (x**2) + 2

punto_fijo_con_rango(g9, rangoInferior=2.5, rangoSuperior=3.0, x0=2.5, iteraciones=30, tolerancia=1e-3)

#------------------------------------------------------------------
# Ejercicio 10
def g10(x):
    return np.sqrt(np.exp(x) / 3)

punto_fijo_con_rango(g10, rangoInferior=0.0, rangoSuperior=1.0, x0=0.5, iteraciones=20)
