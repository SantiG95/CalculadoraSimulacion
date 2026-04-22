import math
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

def biseccion(funcion, inicio, fin, iteraciones=100, tolerancia=1e-6, precision=10):
    inicio_original = inicio
    fin_original = fin
    
    # Evaluamos los extremos al principio
    f_inicio = round(funcion(inicio), precision)
    f_fin = round(funcion(fin), precision)

    # 1. Verificamos si el límite inferior es raíz
    if abs(f_inicio) <= tolerancia:
        print(f"¡Atención! El límite inferior a = {inicio} es una raíz.")
        graficar_biseccion(funcion, inicio_original, fin_original, inicio)
        return inicio
        
    # 2. Verificamos si el límite superior es raíz
    if abs(f_fin) <= tolerancia:
        print(f"¡Atención! El límite superior b = {fin} es una raíz.")
        graficar_biseccion(funcion, inicio_original, fin_original, fin)
        return fin

    # 3. Comprobamos que haya cambio de signo para aplicar el método
    if f_inicio * f_fin > 0:
        raise ValueError(f"La función no cambia de signo en [{inicio}, {fin}]. f(a)={f_inicio}, f(b)={f_fin}")
    resultados = []
    raiz_encontrada = None
    
    for i in range(iteraciones):
        c = round((inicio + fin) / 2.0, precision)
        fc = round(funcion(c), precision)
        er = round((fin - inicio) / 2.0, 6)
        resultados.append([i+1, inicio, fin, c, fc, er])

        # Si encontramos la raíz, guardamos el valor y rompemos el bucle
        if abs(fc) < tolerancia or (fin - inicio) / 2.0 < tolerancia:
            raiz_encontrada = c
            break
        
        if funcion(inicio) * funcion(c) < 0:
            fin = c
        else:
            inicio = c
            
    headers = ["Iteración", "a", "b", "m (punto medio)", "f(m)", "Error (b-a)/2"]
    print(tabulate(resultados, headers=headers, tablefmt="grid"))
    print()
    
    if raiz_encontrada is not None:
        graficar_biseccion(funcion, inicio_original, fin_original, raiz_encontrada)
        
        return raiz_encontrada
    else:
        raise ValueError("El metodo no convergio")



def graficar_biseccion(f, a, b, raiz):
    # 1. Creamos el rango de X con un pequeño margen para ver mejor
    margen = (b - a) * 0.2
    x = np.linspace(a - margen, b + margen, 1000)
    
    # Manejamos posibles errores de división por cero al graficar
    with np.errstate(divide='ignore', invalid='ignore'):
        y = f(x)

    plt.figure(figsize=(10, 6))
    
    # 2. Graficar la función principal
    plt.plot(x, y, label='$f(x)$', color='blue', linewidth=2)
    
    # 3. Resaltar el intervalo de búsqueda [a, b]
    plt.axvspan(a, b, color='green', alpha=0.1, label='Intervalo $[a, b]$')
    plt.axvline(a, color='green', linestyle='--', alpha=0.5)
    plt.axvline(b, color='green', linestyle='--', alpha=0.5)

    # 4. Marcar la raíz encontrada
    if raiz is not None:
        plt.scatter(raiz, f(raiz), color='red', zorder=5, 
                    label=f'Raíz approx: {raiz:.5f}')
        plt.annotate(f'  Raíz ({raiz:.4f})', (raiz, f(raiz)), color='red', fontweight='bold')

    # 5. Estética y ejes
    plt.axhline(0, color='black', linewidth=1) # Eje X
    plt.axvline(0, color='black', linewidth=1) # Eje Y
    plt.title('Método de Bisección - Localización de Raíz', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.show()


#biseccion(lambda x: math.e**(x)-2-x, 1,2)