import numpy as np
import math
from Metodos import Derivada as md 

# ---- Módulo 1: función continua ----

# Funciones
def f(x): 
    return math.sin(x)

def df_ex(x):  
    return math.cos(x)

# Pasos o h
h = 2.1

# Puntos a evaluar
puntos = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

md.tabla_continua(
    f, puntos, h,
    exact_df       = df_ex,
    mostrar_fx         = True,
    mostrar_central    = True,
    mostrar_segunda    = True,
    mostrar_progresiva = True,
    mostrar_regresiva  = True,
    mostrar_exacta     = True,
    mostrar_error      = True,
)

# ---- Módulo 2: datos discretos ----
tiempos    = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
posiciones = [0, 1.2, 2.1, 2.8, 3.2, 3.4, 3.5]

md.tabla_discreta(tiempos, posiciones, titulo="Movimiento — datos discretos")