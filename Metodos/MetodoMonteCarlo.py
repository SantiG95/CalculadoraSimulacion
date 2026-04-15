import numpy as np
import scipy.stats as stats 
import matplotlib.pyplot as plt

# =====================================================================
# 1. MUESTREO POR RECHAZO (Para Ejercicios 1 y 10)
# =====================================================================

def montecarlo_rechazo_pi(n=10000, hacerGrafico=True, seed=None):
    """Resuelve el Ejercicio 1: Aproximar PI con un cuadrado de lado 2"""
    if seed is not None:
        np.random.seed(seed)
        
    # Cuadrado de lado 2 centrado en el origen: x in [-1, 1], y in [-1, 1]
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    
    # Condición de éxito: caer dentro del círculo de radio 1 (x^2 + y^2 <= 1)
    distancia_cuadrada = x**2 + y**2
    puntos_dentro = distancia_cuadrada <= 1
    exitos = np.sum(puntos_dentro)
    
    # Area cuadrado = 4. Area círculo = pi*r^2 = pi. 
    # Pi = Area Cuadrado * (Exitos / N)
    pi_estimado = 4 * (exitos / n)
    
    print(f"\n--- Ejercicio 1: Aproximación de PI (Rechazo) ---")
    print(f"Puntos totales: {n} | Éxitos: {exitos}")
    print(f"PI Estimado: {pi_estimado:.8f} (Real: {np.pi:.8f})")
    
    if hacerGrafico:
        plt.figure(figsize=(6,6))
        plt.scatter(x[~puntos_dentro], y[~puntos_dentro], color='red', s=1, alpha=0.5, label='Rechazados')
        plt.scatter(x[puntos_dentro], y[puntos_dentro], color='blue', s=1, alpha=0.5, label='Aceptados (Éxito)')
        plt.title(f"Monte Carlo Pi: {pi_estimado:.8f}")
        plt.legend()
        plt.show()
    return pi_estimado

def montecarlo_rechazo_curvas(f_techo, f_piso, a, b, y_min, y_max, n=10000, hacerGrafico=True, seed=None):
    """Resuelve el Ejercicio 10: Área entre dos curvas"""
    if seed is not None:
        np.random.seed(seed)
        
    x = np.random.uniform(a, b, n)
    y = np.random.uniform(y_min, y_max, n)
    
    # Éxito: el punto Y cae por debajo de la curva techo y por encima de la curva piso
    exitos = (y <= f_techo(x)) & (y >= f_piso(x))
    n_exitos = np.sum(exitos)
    
    area_rectangulo = (b - a) * (y_max - y_min)
    area_estimada = area_rectangulo * (n_exitos / n)
    
    print(f"\n--- Ejercicio 10: Área entre curvas (Rechazo) ---")
    print(f"Área Estimada: {area_estimada:.8f}")
    
    if hacerGrafico:
        plt.figure(figsize=(8,6))
        x_plot = np.linspace(a, b, 200)
        plt.plot(x_plot, f_techo(x_plot), 'g-', label='f(x) techo')
        plt.plot(x_plot, f_piso(x_plot), 'y-', label='g(x) piso')
        plt.scatter(x[~exitos], y[~exitos], color='red', s=1, alpha=0.3)
        plt.scatter(x[exitos], y[exitos], color='blue', s=1, alpha=0.5)
        plt.title("Muestreo por rechazo entre curvas")
        plt.legend()
        plt.show()
    return area_estimada

# =====================================================================
# 2. CÁLCULO DINÁMICO DE 'N' (Para Ejercicios 2 y 3)
# =====================================================================

def montecarlo_error_maximo(f, a, b, confianza, error_max, n_piloto=1000, seed=None):
    """
    Resuelve Ej 2 y 3: Hace una muestra piloto para despejar la cantidad
    de puntos (N) necesarios para garantizar un error máximo permitido.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Muestra piloto para estimar la desviación estándar (sigma)
    x_piloto = np.random.uniform(a, b, n_piloto)
    y_piloto = f(x_piloto)
    sigma_est = np.std(y_piloto, ddof=1)
    
    # 2. Despejar N de la fórmula del Error Estándar
    valores_z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576, 0.997: 2.968}
    z = valores_z.get(confianza, 1.96)
    
    # Fórmula: N = (Z * sigma * (b-a) / ErrorMax)^2
    n_necesario = int(np.ceil(((z * sigma_est * (b - a)) / error_max)**2))
    
    print(f"\n--- Análisis de Error Máximo ---")
    print(f"Para garantizar un error < {error_max:.8f} al {confianza*100:.8f}%, se necesitan N = {n_necesario} puntos.")
    
    # 3. Ejecutar la integración real con el N calculado
    # Se le pasa la misma semilla si la función integral_monte_carlo la soporta
    "return integral_monte_carlo(f, a, b, confianza, n=n_necesario, hacerGrafico=False)"

# =====================================================================
# 3. SIMULADORES ESPECÍFICOS (Para Ejercicios 11 y 12)
# =====================================================================

def simulador_orbital(dv_nominal, sigma_dv, t_nominal, sigma_t, n=100000, seed=None):
    """
    Resuelve Ej 11: Simula éxito de inserción orbital.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Simulamos el Delta-V y el Tiempo de encendido con distribución normal
    dv_real = np.random.normal(dv_nominal, sigma_dv, n)
    t_real = np.random.normal(t_nominal, sigma_t, n)
    
    # Asumimos una condición de éxito ficticia para el esqueleto del código
    impulso_total = dv_real * t_real
    umbral_exito = (dv_nominal * t_nominal) * 0.95 
    
    exitos = impulso_total >= umbral_exito
    probabilidad = np.mean(exitos)
    
    print(f"\n--- Ejercicio 11: Simulador Orbital ---")
    print(f"Probabilidad de éxito estimada: {probabilidad*100:.8f}%")
    
    # Margen adicional para el 99%
    percentil_1 = np.percentile(impulso_total, 1) # El peor 1% de los casos
    margen = umbral_exito - percentil_1
    print(f"Margen adicional de Delta-V requerido para 99% de éxito: {margen:.8f}")

def simulador_opcion_europea(S0, K, T, r, sigma, n=100000, seed=None):
    """
    Resuelve Ej 12: Precio de Call Europea (Monte Carlo vs Black-Scholes) y VaR.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Monte Carlo
    Z = np.random.normal(0, 1, n)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)
    precio_mc = np.exp(-r * T) * np.mean(payoffs)
    
    # 2. Black-Scholes Analítico (Validación)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    precio_bs = S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    
    # 3. Value at Risk (VaR) 99% a 1 día
    dt = 1/252 # 1 día hábil
    ST_1dia = S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    perdidas = S0 - ST_1dia # Pérdida en la acción
    var_99 = np.percentile(perdidas, 99)
    
    print(f"\n--- Ejercicio 12: Simulación Financiera ---")
    print(f"Precio Call (Monte Carlo): ${precio_mc:.8f}")
    print(f"Precio Call (Black-Scholes): ${precio_bs:.8f}")
    print(f"Diferencia de validación: ${abs(precio_mc - precio_bs):.8f}")
    print(f"VaR 99% a 1 día de la acción: ${var_99:.8f}")
