import numpy as np
from Metodos import EDO  # Importa tu librería original

# ==========================================
# 1. DATOS DE ENTRADA (Modificá esto)
# ==========================================

# Definí la ecuación diferencial: dy/dx = f(x, y)
def f_edo(x, y):
    return x + y 

# Definí la solución exacta (si la tenés) para calcular errores
def f_real(x):
    # Si no la tenés, usá: return np.nan
    return 2 * np.exp(x) - x - 1 

# Parámetros iniciales
x_inicial = 0.0
y_inicial = 1.0
x_final   = 1.0
h_paso    = 0.2  # Tamaño del paso

# ==========================================
# 2. PROCESAMIENTO (No tocar)
# ==========================================

# Cálculo automático de la cantidad de pasos
n_pasos = int(round((x_final - x_inicial) / h_paso))

print(f"Resolviendo EDO desde x={x_inicial} hasta x={x_final}")
print(f"Tamaño de paso h={h_paso} | Total de iteraciones: {n_pasos}\n")

print("=== RESULTADOS: EULER TRADICIONAL ===")
EDO.resolver_edo_euler(f_edo, f_real, x_inicial, y_inicial, h_paso, n_pasos)

print("\n=== RESULTADOS: EULER MEJORADO (HEUN) ===")
EDO.resolver_edo_euler_mejorado(f_edo, f_real, x_inicial, y_inicial, h_paso, n_pasos)

print("\n=== RESULTADOS: RUNGE-KUTTA 4 ===")
EDO.resolver_edo_rk4(f_edo, f_real, x_inicial, y_inicial, h_paso, n_pasos)

print("\nGenerando gráficos comparativos...")
try:
    EDO.graficar_campo_y_solucion(f_edo, f_real, x_inicial, y_inicial, x_final, h_paso, 
                                  titulo="Simulación de EDO - Plantilla")
except TypeError:
    # Por si tu función graficar_campo_y_solucion no acepta el argumento 'titulo'
    EDO.graficar_campo_y_solucion(f_edo, f_real, x_inicial, y_inicial, x_final, h_paso)