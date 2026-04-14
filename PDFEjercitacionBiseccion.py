from Metodos import *
import numpy as np

# 2. Sea 𝑓(𝑥) = 3(𝑥 + 1) (𝑥 − 1/2) (𝑥 − 1) aplique el método de búsqueda binaria de raíces en los siguientes
# intervalos:
# a) [-1, 1.5]
# b) [-1.25, 2.5]

def f1(x):
    return 3 * (x + 1) * (x - 1/2) * (x - 1)

#biseccion(f1, -1, 1.5)
#biseccion(f1, -1.25, 2.5)


# 3. Aplique el método de bisección para encontrar una solución aproximada con tolerancia de 10^(−3), para
# las siguientes funciones en sus intervalos:
# a) √𝑥 − cos(𝑥) = 0, para 0 ≤ 𝑥 ≤ 1
# b) 𝑥 − 2 ^(−𝑥) = 0, para 0 ≤ 𝑥 ≤ 1
# c) 𝑒^𝑥 − 𝑥^2 + 3𝑥 − 2 = 0 para 0 ≤ 𝑥 ≤ 1
# d) 2𝑥 cos(𝑥) − (𝑥 + 1)^2 = 0, para − 3 ≤ 𝑥 ≤ −2, para − 1 ≤ 𝑥 ≤ 0
# e) 𝑥 cos(𝑥) − 2𝑥^2 + 3𝑥 − 1 = 0, para 0.2 ≤ 𝑥 ≤ 0.3, para 1.2 ≤ 𝑥 ≤ 1.3


#print("\nEjercicio A")
#biseccion(lambda x: np.sqrt(x) - np.cos(x), 0, 1, 15, 10**(-3), 6)
#print("\nEjercicio B")
#biseccion(lambda x: x - 2 ** (-x), 0, 1, 15, 10**(-3), 6)
#print("\nEjercicio C")
#biseccion(lambda x: np.e ** x - x ** 2 + 3 * x - 2, 0, 1, 15, 10**(-3), 6)
#print("\nEjercicio D")
#biseccion(lambda x: 2 * x * np.cos(x) - (x + 1) ** 2, -3, -2, 15, 10**(-3), 6)
#biseccion(lambda x: 2 * x * np.cos(x) - (x + 1) ** 2, -1, 0, 15, 10**(-3), 6)
#print("\nEjercicio E")
#biseccion(lambda x: x * np.cos(x) - 2 * x ** 2 + 3 * x - 1, 0.2, 0.3, 15, 10**(-3), 6)
#biseccion(lambda x: x * np.cos(x) - 2 * x ** 2 + 3 * x - 1, 1.2, 1.3, 15, 10**(-3), 6)



# 4. Aplique el método de bisección para encontrar todas las raíces del polinomio dentro de 10^(−2), 
# para 𝑥^4 −2𝑥^3 − 4𝑥^2 + 4𝑥 + 4, en:
# a) [-2, -1]
# b) [0,2]
# c) [2,3]
# d) [-1, 0]

#biseccion(lambda x: x ** 4 - 2 * x ** 3 - 4 * x ** 2 + 4 * x + 4, -2, -1, 15, 10 ** (-2))
#biseccion(lambda x: x ** 4 - 2 * x ** 3 - 4 * x ** 2 + 4 * x + 4, 0, 2, 15, 10 ** (-2))
#biseccion(lambda x: x ** 4 - 2 * x ** 3 - 4 * x ** 2 + 4 * x + 4, 2, 3, 15, 10 ** (-2))
#biseccion(lambda x: x ** 4 - 2 * x ** 3 - 4 * x ** 2 + 4 * x + 4, -1, 0, 15, 10 ** (-2))


# 5. Sea 𝑓(𝑥) = (𝑥 + 2)(𝑥 + 1)(𝑥 − 1)^3 (𝑥 − 2) , 
# ¿a cuál cero la función converge, estudie los siguientes intervalos?
# a) [-3, 2.5]
# b) [-2.5, 3]
# c) [-1.75, 1.5]
# d) [-1.5, 1.75]

#biseccion(lambda x: (x + 2) * (x + 1) * ((x - 1) ** 3) * (x - 2), -3, 2.5, 15, 10**(-3))
#biseccion(lambda x: (x + 2) * (x + 1) * ((x - 1) ** 3) * (x - 2), -2.5, 3, 15, 10**(-3))
#biseccion(lambda x: (x + 2) * (x + 1) * ((x - 1) ** 3) * (x - 2), -1.75, 1.5, 15, 10**(-3))
#biseccion(lambda x: (x + 2) * (x + 1) * ((x - 1) ** 3) * (x - 2), -1.5, 1.75, 15, 10**(-3))


# Use el método del punto fijo para
# 1. 𝑓(𝑥) = 2𝑒^(𝑥^2)− 5𝑥, 𝑥∗𝜖 [0, 1], 𝑥0 = 0
# 2. 𝑓(𝑥) = cos(𝑥), 𝑥 ∗ 𝜖 [1, 2], 𝑥0 = 1
# 3. 𝑓(𝑥) = 𝑒^(−𝑥) − 𝑥, 𝑥∗𝜖 [0, 1], 𝑥0 = 0
# 4. 𝑓(𝑥) = 𝑥^3 − 𝑥 − 1, 𝑥∗𝜖 [1, 2], 𝑥0 = 1
# 5. 𝑓(𝑥) = 𝜋 + 0.5 sin (𝑥/2) − 𝑥, 𝑥∗𝜖 [0, 2𝜋], 𝑥0 = 0

metodo_punto_fijo_con_rango(lambda x: 2 * np.e ** (x ** 2) - 5 * x, 0, 15, 0, 1)