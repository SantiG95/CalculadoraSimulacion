import numpy as np
import scipy.stats as st
import scipy.integrate as integrate

def integrar_montecarlo(func, a, b, confianza, error_max):
    """
    Calcula la integral definida de una función usando el método de Montecarlo.
    
    Parámetros:
    - func: función a integrar (debe poder recibir arrays de numpy)
    - a, b: límites de integración
    - confianza: nivel de confianza (ej. 0.95 para 95%)
    - error_max: error máximo tolerado
    """
    
    # 1. Calcular el valor Z para el nivel de confianza dado
    alfa = 1 - confianza
    z = st.norm.ppf(1 - alfa / 2)
    
    # 2. Muestreo piloto para estimar la desviación estándar de la función
    # Usamos 10,000 muestras iniciales para tener una buena estimación
    N0 = 10000
    x_piloto = np.random.uniform(a, b, N0)
    f_piloto = func(x_piloto)
    desviacion_estandar = np.std(f_piloto, ddof=1)
    
    # 3. Calcular N (número total de iteraciones necesarias)
    # Fórmula del error: Error = z * (b-a) * s / sqrt(N)
    N_requerido = int(np.ceil((z * (b - a) * desviacion_estandar / error_max)**2))
    
    # Nos aseguramos de usar al menos el N calculado o el piloto (el mayor)
    N = max(N_requerido, N0)
    
    # 4. Cálculo final de Montecarlo con el N requerido
    x_final = np.random.uniform(a, b, N)
    f_final = func(x_final)
    
    # I = (b - a) * Promedio(f(x))
    I_montecarlo = (b - a) * np.mean(f_final)
    
    # 5. Calculamos el valor exacto usando cuadratura (para comparar)
    I_exacta, _ = integrate.quad(func, a, b)
    
    return I_montecarlo, I_exacta, N

# ==========================================
# PRUEBA DEL CÓDIGO
# Integral de ln(x) entre 1 y 5
# Confianza: 95%
# Error Máximo: 0.01
# ==========================================

# Fijamos una semilla aleatoria para que el resultado sea reproducible
np.random.seed(42)

funcion = np.log # ln(x) en numpy
lim_inf = 1
lim_sup = 5
nivel_confianza = 0.95
error_maximo = 0.01

I_mc, I_exacta, n_iter = integrar_montecarlo(funcion, lim_inf, lim_sup, nivel_confianza, error_maximo)

print("--- RESULTADOS DE LA INTEGRACIÓN POR MONTECARLO ---")
print(f"I de Montecarlo  : {I_mc:.6f}")
print(f"Valor exacto     : {I_exacta:.6f}")
print(f"Error real       : {abs(I_exacta - I_mc):.6f}")
print(f"N (iteraciones)  : {n_iter:,}")