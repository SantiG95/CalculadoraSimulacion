from Metodos import *
import numpy as np

#metodoMonteCarlo(10)

# integral de e^((-x) ^2) entre 0 y 1
# TODO revisar, da muy alto
print(simpson1_3(lambda x: np.e ** ((-x) ** 2), 0, 1, 100))

# integral de e^((-x) ^2) entre 0 y 1 con nivel de confianza del 99.7%
