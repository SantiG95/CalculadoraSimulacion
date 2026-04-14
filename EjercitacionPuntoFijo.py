from Metodos import *

def g(x):
    return math.pi + 0.5 * math.sin(x / 2)

def f(x):
    return math.pi/2 * x ** 2 - x - 2

def h(x):
    return 2/math.pi + 4/(math.pi * x)

def g2(x):
    return math.e ** (-x)

# Parámetros iniciales
semilla = 1
numero_de_iteraciones = 10 

# Llamada a la función
#metodo_punto_fijo_con_grafico(f, semilla, numero_de_iteraciones)
aceleracion_aitken(g2, semilla, numero_de_iteraciones)
#print("*" * 42)
#metodo_punto_fijo_con_rango(g, semilla, numero_de_iteraciones, 0, 2 * math.pi)

def a(x):
    return x ** 3 - x - 4

def aDerivada(x):
    return 3 * x ** 2 - 1

#newton_raphson(a, aDerivada, semilla, numero_de_iteraciones)

