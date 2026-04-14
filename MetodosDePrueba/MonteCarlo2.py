import random
import math

random.seed(42)

def integral_montecarlo(funcion, a, b, max_y, num_puntos):
    """
    Calcula la integral definida de una función usando el método de Montecarlo (Hit-or-Miss).
    Asume que la función es positiva en el intervalo [a, b].
    """
    aciertos = 0
    
    for _ in range(num_puntos):
        # Generar un punto aleatorio dentro de la caja delimitadora
        x = random.uniform(a, b)
        y = random.uniform(0, max_y)
        
        # Verificar si el punto está por debajo de la curva
        if y <= funcion(x):
            aciertos += 1
            
    # Calcular el área
    area_caja = (b - a) * max_y
    proporcion = aciertos / num_puntos
    area_estimada = area_caja * proporcion
    
    return area_estimada

if __name__ == "__main__":
    print("="*50)
    print("CALCULADORA DE INTEGRALES POR MONTECARLO")
    print("="*50)

    # EJEMPLO 1: Integral de f(x) = x^2 desde 0 hasta 2
    # El valor máximo de x^2 en [0,2] es 4, así que usamos max_y = 4
    func_1 = lambda x: x**2
    resultado_1 = integral_montecarlo(func_1, a=0, b=2, max_y=4, num_puntos=50000)
    print(f"1. Integral de x^2 en [0, 2]:")
    print(f"   Estimación: {resultado_1:.4f} (Valor real: ~2.6667)\n")

    # EJEMPLO 2: Integral de f(x) = sin(x) desde 0 hasta pi
    # El valor máximo de sin(x) es 1, usamos max_y = 1
    func_2 = lambda x: math.sin(x)
    resultado_2 = integral_montecarlo(func_2, a=0, b=math.pi, max_y=1, num_puntos=50000)
    print(f"2. Integral de sin(x) en [0, π]:")
    print(f"   Estimación: {resultado_2:.4f} (Valor real: 2.0000)\n")

    # EJEMPLO 3: Integral de una función más compleja f(x) = e^(-x^2) de 0 a 1
    # Su valor máximo en ese tramo es 1 (cuando x=0)
    func_3 = lambda x: math.exp(-(x**2))
    resultado_3 = integral_montecarlo(func_3, a=0, b=1, max_y=1, num_puntos=50000)
    print(f"3. Integral de e^(-x^2) en [0, 1]:")
    print(f"   Estimación: {resultado_3:.4f} (Valor real: ~0.7468)\n")

    func_4 = lambda x: math.e ** (-x ** 2)
    resultado_4 = integral_montecarlo(func_4, a=0, b=math.pi, max_y=1, num_puntos=10000)
    print(f"   Estimación: {resultado_4:.4f} (Valor real: ~0.7468)\n")