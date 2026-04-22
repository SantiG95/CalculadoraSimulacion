import math
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. MÉTODO DE EULER TRADICIONAL
# ==========================================
def resolver_edo_euler(f_edo, f_real, x0, y0, h, pasos):
    print(f"{'n':<4} | {'Xn':<6} | {'Yn':<9} | {'Yn+1':<9} | {'Yr':<9} | {'E(error)':<9}")
    print("-" * 62)
    
    xn = x0
    yn = y0
    
    for n in range(pasos):
        yr = f_real(xn)
        error = abs(yn - yr)
        yn_mas_1 = yn + h * f_edo(xn, yn)
        print(f"{n:<4} | {xn:<6.2f} | {yn:<9.5f} | {yn_mas_1:<9.5f} | {yr:<9.5f} | {error:<9.5f}")
        yn = yn_mas_1
        xn = xn + h

    yr = f_real(xn)
    error = abs(yn - yr)
    yn_mas_1 = yn + h * f_edo(xn, yn)
    print(f"{pasos:<4} | {xn:<6.2f} | {yn:<9.5f} | {yn_mas_1:<9.5f} | {yr:<9.5f} | {error:<9.5f}")

# ==========================================
# 2. MÉTODO DE EULER MEJORADO (HEUN)
# ==========================================
def resolver_edo_euler_mejorado(f_edo, f_real, x0, y0, h, pasos):
    print(f"{'n':<4} | {'Xn':<6} | {'Yn':<9} | {'Y*n+1':<9} | {'Yn+1':<9} | {'E':<9}")
    print("-" * 62)
    
    xn = x0
    yn = y0
    
    for n in range(pasos):
        yr = f_real(xn)
        error = abs(yn - yr)
        
        k1 = f_edo(xn, yn)
        y_pred = yn + h * k1  
        
        xn_mas_1 = xn + h
        k2 = f_edo(xn_mas_1, y_pred)
        yn_mas_1 = yn + (h / 2) * (k1 + k2)
        
        print(f"{n:<4} | {xn:<6.2f} | {yn:<9.5f} | {y_pred:<9.5f} | {yn_mas_1:<9.5f} | {error:<9.5f}")
        
        yn = yn_mas_1
        xn = xn_mas_1

    yr = f_real(xn)
    error = abs(yn - yr)
    k1 = f_edo(xn, yn)
    y_pred = yn + h * k1
    xn_mas_1 = xn + h
    k2 = f_edo(xn_mas_1, y_pred)
    yn_mas_1 = yn + (h / 2) * (k1 + k2)
    print(f"{pasos:<4} | {xn:<6.2f} | {yn:<9.5f} | {y_pred:<9.5f} | {yn_mas_1:<9.5f} | {error:<9.5f}")

# ==========================================
# 3. MÉTODO DE RUNGE-KUTTA 4 (RK4)
# ==========================================
def resolver_edo_rk4(f_edo, f_real, x0, y0, h, pasos):
    print(f"{'n':<4} | {'Xn':<6} | {'Yn':<9} | {'K1':<9} | {'K2':<9} | {'K3':<9} | {'K4':<9} | {'Yn+1':<9} | {'Yr':<9} | {'E(error)':<9}")
    print("-" * 105)
    
    xn = x0
    yn = y0
    
    for n in range(pasos):
        yr = f_real(xn)
        error = abs(yn - yr)
        
        k1 = f_edo(xn, yn)
        k2 = f_edo(xn + h/2, yn + (h/2) * k1)
        k3 = f_edo(xn + h/2, yn + (h/2) * k2)
        k4 = f_edo(xn + h, yn + h * k3)
        
        yn_mas_1 = yn + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        print(f"{n:<4} | {xn:<6.2f} | {yn:<9.5f} | {k1:<9.5f} | {k2:<9.5f} | {k3:<9.5f} | {k4:<9.5f} | {yn_mas_1:<9.5f} | {yr:<9.5f} | {error:<9.5f}")
        
        yn = yn_mas_1
        xn = xn + h

    yr = f_real(xn)
    error = abs(yn - yr)
    k1 = f_edo(xn, yn)
    k2 = f_edo(xn + h/2, yn + (h/2) * k1)
    k3 = f_edo(xn + h/2, yn + (h/2) * k2)
    k4 = f_edo(xn + h, yn + h * k3)
    yn_mas_1 = yn + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    print(f"{pasos:<4} | {xn:<6.2f} | {yn:<9.5f} | {k1:<9.5f} | {k2:<9.5f} | {k3:<9.5f} | {k4:<9.5f} | {yn_mas_1:<9.5f} | {yr:<9.5f} | {error:<9.5f}")

# ==========================================
# 4. GRÁFICOS: CAMPOS DIRECTORES Y SOLUCIÓN
# ==========================================
def graficar_campo_y_solucion(f_edo, f_real, x0, y0, x_final, h, titulo="Campo Director y Solución Exacta"):
    x_val = np.linspace(x0 - 0.5, x_final + 0.5, 20)
    y_val = np.linspace(y0 - 2, y0 + 5, 20)
    X, Y = np.meshgrid(x_val, y_val)
    
    U = 1 
    V = f_edo(X, Y) 
    
    N = np.sqrt(U**2 + V**2)
    U2, V2 = U/N, V/N
    
    plt.figure(figsize=(10, 6))
    plt.quiver(X, Y, U2, V2, color='lightgray', angles='xy')
    
    x_exacta = np.linspace(x0, x_final, 100)
    y_exacta = [f_real(xi) for xi in x_exacta]
    
    # Manejo de casos como la Ecuación de Riccati (Ejercicio 4) sin solución analítica
    if not np.isnan(y_exacta[0]):
        plt.plot(x_exacta, y_exacta, 'r-', linewidth=2, label='Solución Exacta Analítica')
    else:
        plt.plot([], [], 'r-', label='(Sin Solución Analítica - Riccati)')
        
    plt.plot(x0, y0, 'bo', label='Condición Inicial (x0, y0)')
    
    plt.title(titulo)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()

# ==========================================
# DEFINICIÓN DEL EJERCICIO
# ==========================================
def mi_edo(x, y):
    return x**2 + y

def solucion_real(x):
    return -x - 1 + 2 * math.exp(x)

x_inicial = 0.0
y_inicial = 1.0
tamano_paso = 0.2   
numero_pasos = 10   

x_final = x_inicial + (tamano_paso * numero_pasos)

# ==========================================
# EJECUCIÓN Y COMPARACIÓN
# ==========================================
print("=== EJECUCIÓN: MÉTODO DE EULER TRADICIONAL ===")
#resolver_edo_euler(mi_edo, solucion_real, x_inicial, y_inicial, tamano_paso, numero_pasos)

print("\n\n=== EJECUCIÓN: MÉTODO DE EULER MEJORADO ===")
#resolver_edo_euler_mejorado(mi_edo, solucion_real, x_inicial, y_inicial, tamano_paso, numero_pasos)

print("\n\n=== EJECUCIÓN: MÉTODO DE RUNGE-KUTTA 4 ===")
#resolver_edo_rk4(mi_edo, solucion_real, x_inicial, y_inicial, tamano_paso, numero_pasos)

print("\n=== GENERANDO GRÁFICO ===")
#graficar_campo_y_solucion(mi_edo, solucion_real, x_inicial, y_inicial, x_final, tamano_paso)