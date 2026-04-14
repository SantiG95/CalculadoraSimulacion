def primera_derivada(funcion, punto, paso=1e-3):
    resultado = (funcion(punto + paso) - funcion( punto - paso)) / (2 * paso)
    print(f"Primera derivada en x = {punto}: {resultado}")
    return resultado

def segunda_derivada(funcion, punto, paso=1e-3):
    resultado = (funcion(punto + paso) - 2*funcion(punto) + funcion( punto - paso)) / (paso**2)
    print(f"Segunda derivada en x = {punto}: {resultado}")
    return resultado

def cuarta_derivada(funcion, punto, paso=1e-3):
    # Fórmula de diferencias finitas centrales para la 4ta derivada
    numerador = (funcion(punto + 2*paso) - 4*funcion(punto + paso) + 
                 6*funcion(punto) - 4*funcion(punto - paso) + funcion(punto - 2*paso))
    resultado = numerador / (paso**4)
    return resultado
