import math
import matplotlib.pyplot as plt


def metodo_punto_fijo(funcion, x0, iteraciones):
    print(f"{'n':<5} | {'Xn':<15} | {'X{n+1}':<15}")
    print("-" * 42)
    
    xn = x0
    for n in range(iteraciones):
        xSiguiente = funcion(xn)
        
        print(f"{n:<5} | {xn:<15.8f} | {xSiguiente:<15.8f}")
        
        xn = xSiguiente

def metodo_punto_fijo_con_rango(funcion, x0, iteraciones, rangoInferior, rangoSuperior):
    print(f"{'n':<5} | {'Xn':<15} | {'X{n+1}':<15}")
    print("-" * 42)
    
    xn = x0
    for n in range(iteraciones):
        if not (rangoInferior <= xn <= rangoSuperior):
            print(f"Alerta: Xn ({xn:.4f}) salio del intervalo")
            break
        xSiguiente = funcion(xn)
        
        print(f"{n:<5} | {xn:<15.8f} | {xSiguiente:<15.8f}")
        
        xn = xSiguiente

def metodo_punto_fijo_con_grafico(funcion, x0, iteraciones, hacerGrafico=True):
    print(f"{'n':<5} | {'Xn':<15} | {'X{n+1}':<15}")
    print("-" * 42)
    
    xn = x0

    valores_n = []
    valores_x = []

    for n in range(iteraciones):
        xSiguiente = funcion(xn)

        valores_n.append(n)
        valores_x.append(xSiguiente)
        
        print(f"{n:<5} | {xn:<15.8f} | {xSiguiente:<15.8f}")
        
        xn = xSiguiente

    valores_n.append(iteraciones)
    valores_x.append(xn)

    if hacerGrafico:
        realizar_grafico(valores_n, valores_x, xn)

def realizar_grafico(valores_n, valores_x, xn):
    plt.figure(figsize=(8, 5))

    plt.plot(valores_n, valores_x, marker='o', linestyle='-', color='b', label='X_n (Aproximación)')
    
    plt.axhline(y=xn, color='r', linestyle='--', label='Punto fijo (Límite)')
    
    plt.title('Convergencia del Método de Punto Fijo')
    plt.xlabel('Iteración (n)')
    plt.ylabel('Valor de X_n')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Muestra el gráfico en una ventana
    plt.show()