#TODO Prioridad para el examen

import math
import matplotlib.pyplot as plt

def newton_raphson(funcion, funcionDerivada, x0, iteraciones, hacerGrafico=True):
    print(f"{'n':<5} | {'Xn':<12} | {'F(Xn)':<12} | {"F'(n+1)":<12} | {'X(n+1)':<12}")
    print("-" * 65)

    valores_n = []
    valores_x = []

    xn = x0
    for n in range(iteraciones):
        f = funcion(xn)
        fDerivada = funcionDerivada(xn)

        if fDerivada == 0:
            print(f"\nDerivada nula en la iteración {n}. No se puede dividir por cero.")
            break

        xSiguiente = aceleracionNewton(xn, f, fDerivada)

        valores_n.append(n)
        valores_x.append(xSiguiente)
        
        print(f"{n:<5} | {xn:.10f} | {f:.10f} | {fDerivada:.10f} | {xSiguiente:.10f}")
        
        if xn == xSiguiente:
            print(f"\nConvergencia máxima alcanzada en la iteración {n}.")
            break

        xn = xSiguiente

    if hacerGrafico and len(valores_n) > 0:
        realizar_grafico(valores_n, valores_x, xn)


def aceleracionNewton(xn, fXn, fDerivadaXn):
        return xn - fXn / fDerivadaXn

def realizar_grafico(valores_n, valores_x, xn):
    plt.figure(figsize=(8, 5))

    plt.plot(valores_n, valores_x, marker='o', linestyle='-', color='b', label='X_n (Aproximación)')
    
    plt.axhline(y=xn, color='r', linestyle='--', label=f'Raíz encontrada ({xn:.6f})')

    plt.title('Convergencia del Método de Newton-Raphson')
    plt.xlabel('Iteración (n)')
    plt.ylabel('Valor de X_n')
    plt.xticks(valores_n)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Muestra el gráfico en una ventana
    plt.show()