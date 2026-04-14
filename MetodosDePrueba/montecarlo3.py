import random
import math

def integral_montecarlo(funcion, a, b, num_puntos, confianza=0.95):
    valores_z = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576
    }
    
    if confianza not in valores_z:
        raise ValueError("Por favor, usa un nivel de confianza de 0.90, 0.95 o 0.99")
    
    z = valores_z[confianza]
    
    # Generar puntos y evaluar la función
    x_aleatorios = [random.uniform(a, b) for _ in range(num_puntos)]
    y_evaluaciones = [funcion(x) * (b - a) for x in x_aleatorios]
    
    # Cálculos estadísticos
    i_montecarlo = sum(y_evaluaciones) / num_puntos # Esta es la I de Montecarlo
    varianza = sum((y - i_montecarlo)**2 for y in y_evaluaciones) / (num_puntos - 1)
    desviacion_estandar = math.sqrt(varianza)
    error_estandar = desviacion_estandar / math.sqrt(num_puntos)
    
    # Intervalo de Confianza
    margen_error = z * error_estandar
    limite_inf = i_montecarlo - margen_error
    limite_sup = i_montecarlo + margen_error
    
    # Mostrar tabla de valores
    print("--- Tabla de Valores (Muestra de 10 puntos) ---")
    print(f"{'Punto x':<15} | {'f(x) * (b-a)':<15}")
    print("-" * 35)
    for i in range(min(10, num_puntos)):
        print(f"{x_aleatorios[i]:<15.6f} | {y_evaluaciones[i]:<15.6f}")
        
    # Mostrar el Análisis Estadístico
    print("\n" + "="*35)
    print("      ANÁLISIS ESTADÍSTICO")
    print("="*35)
    print(f"Muestras (N)        : {num_puntos}")
    print(f"Nivel de Confianza  : {confianza * 100}% (Z = {z})")
    print("-" * 35)
    print(f"I de Montecarlo     : {i_montecarlo:.6f}") # <- Etiqueta actualizada
    print("-" * 35)
    print(f"Desviación Estándar : {desviacion_estandar:.6f}")
    print(f"Error Estándar      : {error_estandar:.6f}")
    print(f"Intervalo Confianza : [{limite_inf:.6f}, {limite_sup:.6f}]")
    print("="*35 + "\n")
    
    return i_montecarlo, (limite_inf, limite_sup)

# ==========================================
# EJEMPLO DE USO
# ==========================================
if __name__ == "__main__":
    def mi_funcion(x):
        return math.log(x)

    print("Integrando f(x) = x^2 en el intervalo [0, 2]\n")
    integral_montecarlo(mi_funcion, a=1, b=5, num_puntos=10000, confianza=0.95)