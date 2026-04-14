#TODO Prioridad para el examen

import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def aceleracion_aitken(funcion, x0, iteraciones, tol=1e-6, hacerGrafico=True):
    # Lista para guardar los datos de la tabla
    resultados_tabla = []
    
    # Listas para el gráfico
    valores_n = []
    valores_x = []

    xn = x0
    convergio = False

    for n in range(iteraciones):
        # 1. Calculamos los dos pasos siguientes de punto fijo
        x_sig = funcion(xn)
        x_sig_sig = funcion(x_sig)
        # 2. Denominador de Aitken
        denominador = x_sig_sig - 2 * x_sig + xn
        if denominador == 0:
            # Si el denominador es 0, usamos el valor simple para no romper el script
            xAcelerado = x_sig_sig
        else:
            # Aplicamos la fórmula de aceleración
            xAcelerado = aceleracion(xn, x_sig, x_sig_sig)
        # 3. Calculamos el error absoluto
        error = abs(xAcelerado - xn)
        # Guardamos datos para la tabla y el gráfico
        resultados_tabla.append([n, f"{xn:.10f}", f"{x_sig:.10f}", f"{x_sig_sig:.10f}", f"{xAcelerado:.10f}", f"{error:.2e}"])
        valores_n.append(n)
        valores_x.append(xAcelerado)
        # 4. Verificamos la TOLERANCIA
        if error < tol:
            print(f"\n✅ Convergencia alcanzada en la iteración {n} con tolerancia {tol}")
            convergio = True
            xn = xAcelerado # Actualizamos al último valor antes de salir
            break
        # Actualizamos para la siguiente iteración
        xn = xAcelerado

    # Imprimimos la tabla usando tabulate
    headers = ["n", "Xn", "X(n+1)", "X(n+2)", "Xn* (Acelerado)", "Error Abs."]
    print(tabulate(resultados_tabla, headers=headers, tablefmt="grid"))

    if not convergio:
        print(f"\n⚠️ El método alcanzó las {iteraciones} iteraciones sin llegar a la tolerancia.")

    if hacerGrafico and len(valores_n) > 0:
        realizar_grafico(valores_n, valores_x, xn)
    return xn

def aceleracion(xn, xn1, xn2):
    return xn - ((xn1 - xn)**2) / (xn2 - 2 * xn1 + xn)

def realizar_grafico(valores_n, valores_x, xn):
    plt.figure(figsize=(10, 5))
    plt.plot(valores_n, valores_x, marker='o', linestyle='-', color='b', label='$\hat{X}_n$ (Aitken)')
    plt.axhline(y=xn, color='r', linestyle='--', label=f'Raíz approx: {xn:.8f}')
    
    plt.title('Convergencia del Método de Aitken ($\Delta^2$)', fontsize=12)
    plt.xlabel('Iteración (n)')
    plt.ylabel('Valor de la raíz')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.show()