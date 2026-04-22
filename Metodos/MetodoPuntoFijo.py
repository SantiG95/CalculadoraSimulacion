import math
import matplotlib.pyplot as plt

def punto_fijo(funcion, x0, iteraciones, hacerGrafico=True, tolerancia=1e-4):
    'Realiza el método de punto fijo para encontrar una raíz de la función dada.'
    print(f"{'n':<5} | {'Xn':<15} | {'X{n+1}':<15} | {'Error':<15}")
    print("-" * 58)
    
    xn = x0
    valores_n = []
    valores_x = []
    raiz_encontrada = None

    for n in range(iteraciones):
        xSiguiente = funcion(xn)
        error = abs(xSiguiente - xn)

        valores_n.append(n)
        valores_x.append(xSiguiente)
        
        print(f"{n:<5} | {xn:<15.8f} | {xSiguiente:<15.8f} | {error:<15.8f}")
        
        if error < tolerancia:
            raiz_encontrada = xSiguiente
            print(f"\nConvergencia alcanzada en la iteración {n}.")
            break
            
        xn = xSiguiente

    if raiz_encontrada is None:
        valores_n.append(iteraciones)
        valores_x.append(xn)
        print(f"\nEl método no alcanzó la tolerancia en {iteraciones} iteraciones.")

    if hacerGrafico:
        realizar_grafico(valores_n, valores_x, xSiguiente)
        
    return xSiguiente

def punto_fijo_con_rango(funcion, rangoInferior, rangoSuperior ,x0 , iteraciones = 20, tolerancia=1e-4):
    'Realiza el método de punto fijo con un rango específico para las iteraciones.'
    print(f"{'n':<5} | {'Xn':<15} | {'X{n+1}':<15} | {'Error':<15}")
    print("-" * 58)
    
    xn = x0
    raiz_encontrada = None

    for n in range(iteraciones):
        if not (rangoInferior <= xn <= rangoSuperior):
            print(f"\nAlerta: Xn ({xn:.4f}) salió del intervalo [{rangoInferior}, {rangoSuperior}]")
            break
            
        xSiguiente = funcion(xn)
        error = abs(xSiguiente - xn)
        
        print(f"{n:<5} | {xn:<15.8f} | {xSiguiente:<15.8f} | {error:<15.8f}")
        
        if error < tolerancia:
            raiz_encontrada = xSiguiente
            print(f"\nConvergencia alcanzada en la iteración {n}.")
            break
            
        xn = xSiguiente

    if raiz_encontrada is None and (rangoInferior <= xn <= rangoSuperior):
         print(f"\nEl método no alcanzó la tolerancia en {iteraciones} iteraciones.")
         
    return xSiguiente

def realizar_grafico(valores_n, valores_x, x_final):
    plt.figure(figsize=(8, 5))

    plt.plot(valores_n, valores_x, marker='o', linestyle='-', color='b', label='$X_n$ (Aproximación)')
    
    plt.axhline(y=x_final, color='r', linestyle='--', label=f'Punto fijo aprox: {x_final:.5f}')
    
    plt.title('Convergencia del Método de Punto Fijo')
    plt.xlabel('Iteración (n)')
    plt.ylabel('Valor de $X_n$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.show()