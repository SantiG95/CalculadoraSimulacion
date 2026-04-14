from Metodos import *
import math
import os
os.system("cls")

#newton_raphson(a, aDerivada, semilla, numero_de_iteraciones)

def f1(x):
    return (x-1)**2

def f1Derivada(x):
    return 2 * (x-1)

semilla = 0
#newton_raphson(f1, f1Derivada, semilla, 10)


def f2(x):
    return x**6 - 2

def f2Derivada(x):
    return 6 * x ** 5

semilla = 1

#newton_raphson(f2, f2Derivada, semilla, 10)

def f3(x):
    return math.e ** x + x ** 2 - 4

def f3Derivada(x):
    return math.e ** x + 2 * x

semilla = 0.5

newton_raphson(f3, f3Derivada, semilla, 10)

def f4(x): return math.log(x) -1

def f4Derivada(x): return 1/x

semilla = 2

#newton_raphson(f4, f4Derivada, semilla, 10, False)

