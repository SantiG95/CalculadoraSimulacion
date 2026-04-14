import numpy as np
import scipy.stats as stats
import pandas as pd

np.random.seed(0)

def integracion_montecarlo(funcion, a, b, nivel_confianza, error_maximo, muestras_iniciales=10000):
    """
    Calcula la integral de una función usando Montecarlo, ajustando la cantidad de
    muestras para cumplir con un error máximo dado.
    """
    print(f"--- Iniciando Integración de Montecarlo ---")
    print(f"Límites: [{a}, {b}], Confianza: {nivel_confianza*100}%, Error Máx: {error_maximo}")
    
    # 1. Calcular el valor Z (z-score) para el intervalo de confianza
    alpha = 1 - nivel_confianza
    z_score = stats.norm.ppf(1 - alpha/2)
    
    # 2. Lote inicial para estimar la desviación estándar de la función
    # Esto es necesario para saber cuántas muestras necesitaremos en total
    x_inicial = np.random.uniform(a, b, muestras_iniciales)
    y_inicial = funcion(x_inicial)
    desviacion_std_inicial = np.std(y_inicial, ddof=1)
    
    # 3. Calcular el N requerido para alcanzar el error máximo deseado
    # Fórmula del error: E = Z * (b-a) * (std / sqrt(N))
    # Despejando N: N = (Z * (b-a) * std / E)^2
    N_requerido = int(np.ceil((z_score * (b - a) * desviacion_std_inicial / error_maximo)**2))
    
    # Aseguramos un mínimo de muestras y aplicamos un límite de seguridad razonable
    N_total = max(muestras_iniciales, N_requerido)
    limite_memoria = 100_000_000 # Evita colapsar la RAM si el error pedido es absurdamente pequeño
    
    if N_total > limite_memoria:
        print(f"⚠️ Advertencia: Se requerían {N_total} muestras, pero se limitó a {limite_memoria} por seguridad.")
        N_total = limite_memoria
    else:
        print(f"Calculado: Se generarán {N_total} muestras totales para garantizar el error.")

    # 4. Generar la simulación completa
    x_muestras = np.random.uniform(a, b, N_total)
    y_muestras = funcion(x_muestras)
    
    # 5. Cálculos Estadísticos
    media_y = np.mean(y_muestras)
    estimacion_integral = (b - a) * media_y
    
    desviacion_estandar_y = np.std(y_muestras, ddof=1)
    error_estandar = (b - a) * (desviacion_estandar_y / np.sqrt(N_total))
    
    margen_error_real = z_score * error_estandar
    limite_inferior = estimacion_integral - margen_error_real
    limite_superior = estimacion_integral + margen_error_real
    
    # 6. Crear una tabla de valores representativa (10 puntos distribuidos)
    # No mostramos todas las muestras porque colapsaría la pantalla
    indices_tabla = np.linspace(0, N_total-1, 10, dtype=int)
    tabla_valores = pd.DataFrame({
        'Muestra N°': indices_tabla + 1,
        'x generada': x_muestras[indices_tabla],
        'f(x)': y_muestras[indices_tabla]
    })
    
    # 7. Retornar resultados
    resultados = {
        "Muestras Totales (N)": N_total,
        "Media de f(x)": media_y,
        "Desviación Estándar de f(x)": desviacion_estandar_y,
        "Error Estándar de la Integral": error_estandar,
        "Estimación de la Integral": estimacion_integral,
        "Nivel de Confianza": f"{nivel_confianza * 100}%",
        "Intervalo de Confianza": f"[{limite_inferior:.6f}, {limite_superior:.6f}]",
        "Error Máximo Alcanzado (Margen)": margen_error_real,
        "Tabla de Valores": tabla_valores
    }
    
    return resultados

# ==========================================
# SECCIÓN DE EJECUCIÓN (Ejemplo de uso)
# ==========================================
if __name__ == "__main__":
    # Definir la función a integrar matemáticamente.
    # Ejemplo: f(x) = sin(x) + x^2
    # Utiliza las funciones matemáticas de numpy (np.sin, np.exp, np.log, etc.)
    mi_funcion = lambda x: np.log(x)
    
    # Definir los parámetros requeridos
    limite_a = 1          # Límite inferior de la integral
    limite_b = 5        # Límite superior de la integral
    confianza = 0.95        # Intervalo de confianza (ej. 0.95 para 95%)
    error_max = 0.01       # Error máximo permitido
    
    # Ejecutar la función
    analisis = integracion_montecarlo(
        funcion=mi_funcion, 
        a=limite_a, 
        b=limite_b, 
        nivel_confianza=confianza, 
        error_maximo=error_max
    )
    
    # Imprimir los resultados por pantalla
    print("\n" + "="*40)
    print("ANÁLISIS ESTADÍSTICO Y RESULTADOS")
    print("="*40)
    for clave, valor in analisis.items():
        if clave != "Tabla de Valores":
            print(f"{clave}: {valor}")
            
    print("\n" + "="*40)
    print("TABLA DE VALORES (Muestra representativa de 10 puntos)")
    print("="*40)
    print(analisis["Tabla de Valores"].to_string(index=False))