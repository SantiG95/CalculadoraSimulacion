import scipy.integrate as integrate

def integral_exacta(funcion, inicio, fin):
    """
    Calcula la integral definida de manera "exacta" (con precisión de máquina)
    para usarla como valor real o analítico.
    """
    # integrate.quad devuelve dos valores: el resultado y el margen de error.
    # Solo nos interesa el resultado, por eso usamos [0]
    resultado = integrate.quad(funcion, inicio, fin)[0]
    return resultado