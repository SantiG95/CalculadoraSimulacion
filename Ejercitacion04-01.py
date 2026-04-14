from Metodos import *
import numpy as np

def f1(x):
    return math.e ** (x ** 2)

#print(rectanguloMedioCompuesto(f1, 0, 1, 4))
#print(trapecioCompuesta(f1, 0, 1, 4))
#print(simpson1_3(f1, 0, 1, 4))

def f2(x):
    return 6 + 3 * np.cos(x)

#print(trapecioCompuesta(f2, 0, (math.pi)/2, 4))
#print(simpson1_3(f2, 0, (math.pi)/2, 4))

def f3(x):
    return x ** x

#print(trapecioCompuesta(f3, 0, 1, 4))

def f4(x):
    return np.divide(np.sin(x), x, out=np.ones_like(x, dtype=float), where=x!=0)

#print(trapecioCompuesta(f4, 0, 1, 4))


def f5(x):
    return (4 * x - 3) ** 3

#print(trapecioCompuesta(f5, -3, 3, 6))
#print(simpson1_3(f5, -3, 3, 6))

def f6(x):
    return x ** 2 * np.e ** x

#print(trapecioCompuesta(f6, 0, 3, 6))
#print(simpson1_3(f6, 0, 3, 6))

def f7(x):
    return (1 + x ** 2) ** (1/4)

#print(trapecioCompuesta(f7, 0, 2, 6))
#print(simpson1_3(f7, 0, 2, 6))

def f8(x):
    return np.e ** (x ** 4)

#print(trapecioCompuesta(f8, -1, 1, 5))

def f9(x):
    return np.divide(np.sin(x+1), (1-x)**1/2, out=np.ones_like(x, dtype=float), where=x!=0)

#print(trapecioCompuesta(f9, 0, 1, 6))

def f10(x):
    return np.e ** x

#print(trapecioCompuesta(f10, -np.inf, 0, 6))

def f11(x):
    return (2 + np.cos(1+x**(3/2))/(1 + 0.5 * np.sin(x))**(1/2)) * np.e ** (0.5 * x)

print(simpson1_3(f11, 0, 2, 6))