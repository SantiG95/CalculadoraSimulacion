import math
import random
import statistics


def metodoMonteCarlo(numeroPuntos):
    puntos_dentro = 0
    for _ in range(numeroPuntos):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x**2 + y**2 <= 1:
            puntos_dentro += 1

    pi_estimado = (puntos_dentro / numeroPuntos) * 4
    print(f"Estimacion de pi con {numeroPuntos} puntos: {pi_estimado}")
    return pi_estimado


def _z_score(confidence: float) -> float:
    if confidence == 0.90:
        return 1.6448536269514722
    if confidence == 0.95:
        return 1.959963984540054
    if confidence == 0.99:
        return 2.575829303548901
    return statistics.NormalDist().inv_cdf(0.5 + confidence / 2)


def _area(bounds):
    if len(bounds) == 2:
        a, b = bounds
        return b - a
    if len(bounds) == 4:
        ax, bx, ay, by = bounds
        return (bx - ax) * (by - ay)
    raise ValueError("Los límites deben ser de longitud 2 (1D) o 4 (2D).")


def _integral_referencia_1d(func, a, b, pasos=10000):
    h = (b - a) / pasos
    total = 0.5 * (func(a) + func(b))
    for i in range(1, pasos):
        x = a + i * h
        total += func(x)
    return total * h


def _integral_referencia_2d(func, ax, bx, ay, by, nx=150, ny=150):
    hx = (bx - ax) / nx
    hy = (by - ay) / ny
    total = 0.0
    for i in range(nx + 1):
        x = ax + i * hx
        wx = 1.0 if 0 < i < nx else 0.5
        for j in range(ny + 1):
            y = ay + j * hy
            wy = 1.0 if 0 < j < ny else 0.5
            total += func(x, y) * wx * wy
    return total * hx * hy


def _calcular_valor_referencia(func, bounds):
    if len(bounds) == 2:
        a, b = bounds
        return _integral_referencia_1d(func, a, b)
    ax, bx, ay, by = bounds
    return _integral_referencia_2d(func, ax, bx, ay, by)


def metodoMonteCarloIntegral(func, bounds, numeroPuntos, confidence=0.95, actual_value=None, seed=None):
    if seed is not None:
        random.seed(seed)

    if len(bounds) not in (2, 4):
        raise ValueError("Los límites deben ser de longitud 2 (1D) o 4 (2D).")

    muestras = []
    if len(bounds) == 2:
        a, b = bounds
        for _ in range(numeroPuntos):
            x = random.uniform(a, b)
            muestras.append(func(x))
    else:
        ax, bx, ay, by = bounds
        for _ in range(numeroPuntos):
            x = random.uniform(ax, bx)
            y = random.uniform(ay, by)
            muestras.append(func(x, y))

    media_f = statistics.mean(muestras)
    std_f = statistics.stdev(muestras) if numeroPuntos > 1 else 0.0
    area = _area(bounds)
    estimacion = media_f * area
    stderr = (std_f / math.sqrt(numeroPuntos)) * area
    z = _z_score(confidence)
    intervalo = (estimacion - z * stderr, estimacion + z * stderr)

    if actual_value is None:
        actual_value = _calcular_valor_referencia(func, bounds)

    resultado = {
        "dimension": 1 if len(bounds) == 2 else 2,
        "n_samples": numeroPuntos,
        "bounds": bounds,
        "estimate": estimacion,
        "mean_f": media_f,
        "std_f": std_f,
        "stderr": stderr,
        "confidence": confidence,
        "ci": intervalo,
        "actual_value": actual_value,
        "absolute_error": abs(estimacion - actual_value),
        "samples": muestras if numeroPuntos <= 20 else None,
    }
    return resultado


def imprimir_reporte_integral(resultado):
    print("\nInforme de integración Monte Carlo")
    print("=" * 70)
    print(f"Dimensión: {resultado['dimension']}D")
    print(f"Límites: {resultado['bounds']}")
    print(f"Número de muestras: {resultado['n_samples']}")
    print(f"Valor real (referencia): {resultado['actual_value']:.10f}")
    print(f"Estimación Monte Carlo: {resultado['estimate']:.10f}")
    print(f"Error absoluto: {resultado['absolute_error']:.10f}")
    print(f"Media de f(x): {resultado['mean_f']:.10f}")
    print(f"Desviación estándar de f(x): {resultado['std_f']:.10f}")
    print(f"Error estándar de la integral: {resultado['stderr']:.10f}")
    print(f"Intervalo de confianza {int(resultado['confidence']*100)}%: [{resultado['ci'][0]:.10f}, {resultado['ci'][1]:.10f}]")
    if resultado['samples'] is not None:
        print("\nMuestras de f(x) utilizadas:")
        for i, valor in enumerate(resultado['samples'], start=1):
            print(f"  [{i:2}] = {valor:.10f}")


def metodoMonteCarloIntegralTabla(func, bounds, sample_sizes, actual_value=None, confidence=0.95, seed=None):
    if seed is not None:
        random.seed(seed)

    filas = []
    for n in sample_sizes:
        datos = metodoMonteCarloIntegral(func, bounds, n, confidence, actual_value, None)
        filas.append(datos)

    encabezado = (
        f"{'N':>8} | {'Estimación':>12} | {'Media f':>10} | {'Std f':>10} | {'Err est.':>10} | {'IC inferior':>12} | {'IC superior':>12} | {'Error abs.':>10}"
    )
    print("\nTabla de resultados Monte Carlo")
    print("=" * len(encabezado))
    print(encabezado)
    print("-" * len(encabezado))
    for fila in filas:
        print(
            f"{fila['n_samples']:8d} | {fila['estimate']:12.8f} | {fila['mean_f']:10.6f} | {fila['std_f']:10.6f} | {fila['stderr']:10.6f} | {fila['ci'][0]:12.8f} | {fila['ci'][1]:12.8f} | {fila['absolute_error']:10.6f}"
        )
    print("=" * len(encabezado))
    return filas

f = lambda x: math.e**(-x**2)
metodoMonteCarloIntegralTabla(f, (0, 1), [100, 1000, 5000], actual_value=1/3, seed=123)