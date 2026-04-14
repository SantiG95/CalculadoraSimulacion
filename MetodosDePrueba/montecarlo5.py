import numpy as np
import scipy.stats as stats
import pandas as pd

def montecarlo_adaptativo(func, a, b, nivel_confianza=0.95, error_maximo=0.01, n_inicial=10000):
    """
    Calcula la integral definida de una función usando Montecarlo, garantizando un error máximo.
    """
    # 1. Calcular el valor Z (Z-score) para el nivel de confianza deseado
    alfa = 1 - nivel_confianza
    z_score = stats.norm.ppf(1 - alfa / 2)
    
    # 2. Muestreo inicial para estimar la desviación estándar
    x_inicial = np.random.uniform(a, b, n_inicial)
    y_inicial = func(x_inicial)
    desviacion_estandar_inicial = np.std(y_inicial, ddof=1)
    
    # 3. Calcular la cantidad de muestras 'n' requeridas para cumplir con el error máximo
    # Fórmula: n = (Z * (b-a) * S / E_max)^2
    n_requerido = int(np.ceil((z_score * (b - a) * desviacion_estandar_inicial / error_maximo)**2))
    
    # Usamos el n requerido, pero nunca menos del muestreo inicial
    n_final = max(n_inicial, n_requerido)
    
    # Límite de seguridad para evitar desbordamiento de memoria (RAM)
    if n_final > 5 * 10**7:
        print(f"⚠️ Advertencia: El 'n' requerido ({n_final:,}) es excesivamente alto. Se limitará a 50,000,000 para proteger la memoria.")
        n_final = 50000000

    # 4. Simulación Montecarlo Final
    x_final = np.random.uniform(a, b, n_final)
    y_final = func(x_final)
    
    # 5. Cálculos Estadísticos
    media_y = np.mean(y_final)
    desviacion_estandar_y = np.std(y_final, ddof=1)
    
    estimacion_integral = (b - a) * media_y
    error_estandar = (b - a) * desviacion_estandar_y / np.sqrt(n_final)
    margen_error_real = z_score * error_estandar
    
    limite_inferior = estimacion_integral - margen_error_real
    limite_superior = estimacion_integral + margen_error_real
    
    # 6. Tabular resultados con Pandas
    resultados = {
        "Métrica": [
            "Función Integrada (aprox)",
            "Límite Inferior (a)",
            "Límite Superior (b)",
            "Nivel de Confianza",
            "Error Máximo Permitido",
            "Muestras Calculadas (n)",
            "Media de f(x)",
            "Desviación Estándar de f(x)",
            "Error Estándar (SE)",
            "Margen de Error Obtenido",
            "Intervalo de Confianza",
            "Estimación de la Integral (I)"
        ],
        "Valor": [
            "f(x)",
            a,
            b,
            f"{nivel_confianza * 100}%",
            error_maximo,
            f"{n_final:,}",
            round(media_y, 6),
            round(desviacion_estandar_y, 6),
            round(error_estandar, 6),
            round(margen_error_real, 6),
            f"[{round(limite_inferior, 6)}, {round(limite_superior, 6)}]",
            round(estimacion_integral, 6)
        ]
    }
    
    df_resultados = pd.DataFrame(resultados)
    
    return estimacion_integral, n_final, df_resultados

# ==========================================
# EJEMPLO DE USO
# ==========================================
if __name__ == "__main__":
    # Definir la función a integrar (Ejemplo: f(x) = sin(x) + x^2)
    def mi_funcion(x):
        return np.log(x)

    # Parámetros del usuario
    limite_a = 0
    limite_b = 5
    confianza = 0.95      # 99% de confianza
    error_max = 0.01     # Tolerancia de error muy estricta

    print("Calculando Montecarlo. Por favor espera...\n")
    
    # Ejecutar la función
    I, n_usado, tabla_analisis = montecarlo_adaptativo(
        func = mi_funcion,
        a = limite_a,
        b = limite_b,
        nivel_confianza = confianza,
        error_maximo = error_max
    )
    
    # Imprimir la tabla de análisis estadístico
    print("=== ANÁLISIS ESTADÍSTICO DE MONTECARLO ===")
    print(tabla_analisis.to_string(index=False))
    print("==========================================")