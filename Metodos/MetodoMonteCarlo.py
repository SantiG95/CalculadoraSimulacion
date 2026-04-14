import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
np.random.seed(42)

def integral_monte_carlo(f, a, b, confianza, n=10000, hacerGrafico=True):
    # 1. Generación de puntos y evaluación
    x_aleatorios = np.random.uniform(a, b, n)
    y_evaluados = f(x_aleatorios)
    # 2. Cálculos base
    mean_f = np.mean(y_evaluados)
    resultado = (b - a) * mean_f
    # 3. Estadística (Sigma y Error Estándar)
    sigma = np.std(y_evaluados, ddof=1) # Desviación estándar muestral
    ee = sigma / np.sqrt(n) # Error Estándar 
    # 4. Intervalo de Confianza
    valores_z = {
        0.90: 1.645, 
        0.95: 1.960, 
        0.99: 2.576,
        0.997: 2.968,
        0.999: 3.291
        } # Valores críticos
    z = valores_z.get(confianza, 1.96)
    margen_error = z * ee * (b - a)
    ic_inferior = resultado - margen_error
    ic_superior = resultado + margen_error
    # --- TABLA DE RESULTADOS ACTUALIZADA ---
    datos_resumen = [
        ["Puntos (n)", n],
        ["Promedio f(x)", f"{mean_f:.6f}"],
        ["Desv. Estándar (σ)", f"{sigma:.6f}"], # Agregado a la tabla
        ["Error Estándar (EE)", f"{ee:.6e}"], # Agregado a la tabla
        ["Nivel de Confianza", f"{confianza*100}% (z={z})"],
        ["Margen de Error", f"{margen_error:.6e}"],
        ["IC (Intervalo)", f"[{ic_inferior:.6f}, {ic_superior:.6f}]"]
    ]
    print(f"\n--- Simulación Monte Carlo en [{a}, {b}] ---")
    print(tabulate(datos_resumen, tablefmt="fancy_grid"))
    print(f"RESULTADO DE LA INTEGRAL: {resultado:.10f}\n")
    if hacerGrafico:
        realizar_grafico_monte_carlo(f, a, b, x_aleatorios, y_evaluados)

    return resultado, sigma

def integral_doble_monte_carlo(f, x_min, x_max, y_min, y_max, confianza, n=10000):
    # 1. Muestreo de puntos (x, y) en el rectángulo
    x_random = np.random.uniform(x_min, x_max, n)
    y_random = np.random.uniform(y_min, y_max, n)
    # 2. Evaluación de la función (alturas z)
    z_evaluados = f(x_random, y_random)
    # 3. Cálculos de volumen (Área base * Promedio de alturas)
    mean_z = np.mean(z_evaluados)
    area_base = (x_max - x_min) * (y_max - y_min)
    volumen_estimado = area_base * mean_z
    # 4. Estadística Detallada
    sigma = np.std(z_evaluados, ddof=1) # Desviación estándar de f(xi, yi)
    ee = sigma / np.sqrt(n) # Error Estándar [cite: 167]
    # 5. Intervalo de Confianza (IC)
    valores_z = {
        0.90: 1.645, 
        0.95: 1.960, 
        0.99: 2.576,
        0.997: 2.968,
        0.999: 3.291
        }
    z_critico = valores_z.get(confianza, 1.960)
    # El margen de error se escala por el área de la base para el volumen final
    margen_error = z_critico * ee * area_base
    ic_inf, ic_sup = volumen_estimado - margen_error, volumen_estimado + margen_error
    # --- TABLA DE RESULTADOS ACTUALIZADA ---
    datos_tabla = [
        ["Puntos (n)", n],
        ["Área de la base", f"{area_base:.4f}"],
        ["Promedio f(x,y)", f"{mean_z:.6f}"],
        ["Desv. Estándar (σ)", f"{sigma:.6f}"], # Mide dispersión de la superficie 
        ["Error Estándar (EE)", f"{ee:.6e}"],   # Mide precisión de la media 
        ["Confianza", f"{confianza*100}% (z={z_critico})"],
        ["IC (Intervalo)", f"[{ic_inf:.6f}, {ic_sup:.6f}]"]
    ]
    print(f"\n--- Simulación Monte Carlo Doble ---")
    print(tabulate(datos_tabla, tablefmt="fancy_grid"))
    print(f"VOLUMEN ESTIMADO: {volumen_estimado:.10f}\n")
    
    # Devolvemos el volumen y el desvío para análisis posteriores
    return volumen_estimado, sigma


def realizar_grafico_monte_carlo(f, a, b, x_pts, y_pts):
    plt.figure(figsize=(10, 6))
    # Dibujar la función real (curva suave)
    x_curva = np.linspace(a, b, 500)
    plt.plot(x_curva, f(x_curva), color='red', linewidth=2, label='f(x)')
    # Dibujar una muestra de los puntos (solo 500 para no saturar el gráfico)
    muestra = 500 if len(x_pts) > 500 else len(x_pts)
    plt.scatter(x_pts[:muestra], y_pts[:muestra], color='blue', alpha=0.3, s=10, label=f'Muestra de {muestra} puntos aleatorios')
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Simulación de Monte Carlo (Integración)', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()