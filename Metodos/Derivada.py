import math
import numpy as np

# =========================================================
# MÓDULO 1: FUNCIONES CONTINUAS (Ejercicios 1 al 5)
# =========================================================

def primera_derivada_central(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def segunda_derivada_central(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)

# Fórmulas de orden 2 (Error O(h^2)) exactas para el Ejercicio 5
def primera_derivada_progresiva_o2(f, x, h):
    return (-f(x + 2*h) + 4*f(x + h) - 3*f(x)) / (2 * h)

def primera_derivada_regresiva_o2(f, x, h):
    return (3*f(x) - 4*f(x - h) + f(x - 2*h)) / (2 * h)


# =========================================================
# MÓDULO 2: DATOS DISCRETOS (TABLAS - Ejercicios 6 y 7)
# =========================================================

def derivadas_discretas(t, x):
    """
    Toma arreglos de tiempo (t) y posición (x).
    Devuelve arreglos de velocidad (v) y aceleración (a).
    Usa dif. central en el interior, y progresiva/regresiva simple en los bordes.
    """
    n = len(t)
    v = np.zeros(n)
    a = np.zeros(n)
    
    # 1. Calcular Velocidad
    for i in range(n):
        if i == 0:
            v[i] = (x[i+1] - x[i]) / (t[i+1] - t[i]) # Progresiva
        elif i == n - 1:
            v[i] = (x[i] - x[i-1]) / (t[i] - t[i-1]) # Regresiva
        else:
            v[i] = (x[i+1] - x[i-1]) / (t[i+1] - t[i-1]) # Central
            
    # 2. Calcular Aceleración derivando la velocidad
    for i in range(n):
        if i == 0:
            a[i] = (v[i+1] - v[i]) / (t[i+1] - t[i])
        elif i == n - 1:
            a[i] = (v[i] - v[i-1]) / (t[i] - t[i-1])
        else:
            a[i] = (v[i+1] - v[i-1]) / (t[i+1] - t[i-1])
            
    return v, a

# Ejemplo de uso para la Tabla 1 (Ejercicio 6):
if __name__ == '__main__':
    t_tabla = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    x_tabla = np.array([0, 1.9, 4.2, 7.8, 12, 17, 25, 32, 42])
    
    velocidad, aceleracion = derivadas_discretas(t_tabla, x_tabla)
    
    print("t (s) | x (m) | v (m/s) | a (m/s^2)")
    print("-" * 35)
    for i in range(len(t_tabla)):
        print(f"{t_tabla[i]:<5} | {x_tabla[i]:<5} | {velocidad[i]:<7.3f} | {aceleracion[i]:<7.3f}")