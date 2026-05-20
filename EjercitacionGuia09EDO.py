import numpy as np
from Metodos import EDO  # Importa tu script (asegurate de que el archivo se llame EDO.py y esté en la misma carpeta)

# ==========================================
# DEFINICIÓN DE TODOS LOS EJERCICIOS (1 al 10)
# ==========================================
ejercicios = [
    # --- EJERCICIOS 1 AL 5 ---
#    """{
#        "id": 1,
#        "edo": lambda t, y: y + t**2,
#        "sol_real": lambda t: 3 * np.exp(t) - t**2 - 2*t - 2,
#        "t0": 0.0, "y0": 1.0, "h": 0.1, "t_final": 1.0
#    },
#    {
#        "id": 2,
#        "edo": lambda t, y: y * np.sin(t),
#        "sol_real": lambda t: 2 * np.exp(1 - np.cos(t)),
#        "t0": 0.0, "y0": 2.0, "h": np.pi/10, "t_final": np.pi
#    },
#    {
#        "id": 3,
#        "edo": lambda t, y: 2*t + 3*y,
#        "sol_real": lambda t: (2/9) * (np.exp(3*t) - 3*t - 1),
#        "t0": 0.0, "y0": 0.0, "h": 0.2, "t_final": 1.0
#    },
#    {
#        "id": 4,
#        "edo": lambda t, y: t - y**2,
#        "sol_real": lambda t: np.nan,  # Ecuación de Riccati (sin solución analítica elemental)
#        "t0": 0.0, "y0": 1.0, "h": 0.2, "t_final": 2.0
#    },
#    {
##        "id": 5,
#        "edo": lambda t, y: np.exp(-t) - y,
#        "sol_real": lambda t: t * np.exp(-t),
#        "t0": 0.0, "y0": 0.0, "h": 0.1, "t_final": 1.0
#    },"""
#    
#    # --- EJERCICIOS 6 AL 10 ---
#    """{
#        "id": 6,
#        "edo": lambda t, y: (1 / (1 + t**2)) - y,
#        "sol_real": lambda t: np.nan, # Integral sin solución elemental cerrada
#        "t0": 0.0, "y0": 1.0, "h": 0.5, "t_final": 2.0
#    },
#    {
#        "id": 7,
#        "edo": lambda x, y: (x**2 - 1) / (y**2),
#        "sol_real": lambda x: np.cbrt(x**3 - 3*x + 8),
#        "t0": 0.0, "y0": 2.0, "h": 0.5, "t_final": 2.0
#    },
#    {
#        "id": 8,
#        "edo": lambda t, y: y - t**2 + 1,
#        "sol_real": lambda t: -0.5 * np.exp(t) + t**2 + 2*t + 1,
#        "t0": 0.0, "y0": 0.5, "h": 0.2, "t_final": 1.0
#    },
#    {
#        "id": 9,
#        "edo": lambda x, y: y + x,
#        "sol_real": lambda x: 2 * np.exp(x) - x - 1,
#        "t0": 0.0, "y0": 1.0, "h": 0.2, "t_final": 1.0
#    },
#    {
#        "id": 10,
#        "edo": lambda x, y: y - x,
#        "sol_real": lambda x: -0.5 * np.exp(x) + x + 1,
#        "t0": 0.0, "y0": 0.5, "h": 0.1, "t_final": 1.0
#    }"""
    {
        "id": 11,
        "edo": lambda x, y: 0.1 * np.sqrt(y) + 0.4 * x**2,
        "sol_real": lambda x: np.nan,  # Ecuación no lineal sin solución exacta elemental
        "t0": 2.0, 
        "y0": 2.0, 
        "h": 0.05,  # Calculado a partir de n = 10 en el intervalo [2, 2.5]
        "t_final": 2.5 
    }
]

# ==========================================
# EJECUCIÓN ITERATIVA USANDO TU MÓDULO EDO
# ==========================================
for ej in ejercicios:
    print(f"\n{'='*60}")
    print(f"=== RESOLVIENDO EJERCICIO {ej['id']} ===")
    print(f"{'='*60}")
    
    # Calculamos la cantidad de pasos en base al intervalo y el tamaño de paso (h)
    pasos = int(round((ej["t_final"] - ej["t0"]) / ej["h"]))
    
    print("\n--- MÉTODO DE EULER TRADICIONAL ---")
    EDO.resolver_edo_euler(ej["edo"], ej["sol_real"], ej["t0"], ej["y0"], ej["h"], pasos)
    
    print("\n--- MÉTODO DE EULER MEJORADO ---")
    EDO.resolver_edo_euler_mejorado(ej["edo"], ej["sol_real"], ej["t0"], ej["y0"], ej["h"], pasos)
    
    print("\n--- MÉTODO DE RUNGE-KUTTA 4 ---")
    EDO.resolver_edo_rk4(ej["edo"], ej["sol_real"], ej["t0"], ej["y0"], ej["h"], pasos)
    
    print(f"\nGenerando Gráfico para el Ejercicio {ej['id']}...")
    
    # Manejo del título si actualizaste la función en EDO.py para aceptarlo, si no, sacale el parámetro 'titulo'
    try:
        EDO.graficar_campo_y_solucion(ej["edo"], ej["sol_real"], ej["t0"], ej["y0"], ej["t_final"], ej["h"], titulo=f"Ejercicio {ej['id']}: Campo Director y Solución")
    except TypeError:
        # Por si estás usando la versión original de EDO.py que no tenía el parámetro 'titulo'
        EDO.graficar_campo_y_solucion(ej["edo"], ej["sol_real"], ej["t0"], ej["y0"], ej["t_final"], ej["h"])