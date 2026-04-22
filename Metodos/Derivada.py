import math
import numpy as np
import pandas as pd

# =========================================================
# MÓDULO 1: FUNCIONES CONTINUAS (Ejercicios 1 al 5)
# =========================================================

def primera_derivada_central(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def segunda_derivada_central(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)

def primera_derivada_progresiva_o2(f, x, h):
    return (-f(x + 2*h) + 4*f(x + h) - 3*f(x)) / (2 * h)

def primera_derivada_regresiva_o2(f, x, h):
    return (3*f(x) - 4*f(x - h) + f(x - 2*h)) / (2 * h)

def cuarta_derivada(funcion, punto, paso=1e-3):
    numerador = (funcion(punto + 2*paso) - 4*funcion(punto + paso) +
                 6*funcion(punto) - 4*funcion(punto - paso) + funcion(punto - 2*paso))
    return numerador / (paso**4)


def tabla_continua(f, x_vals, h, exact_df=None, titulo="Tabla de Derivadas",
                   mostrar_fx=True,
                   mostrar_central=False,
                   mostrar_segunda=False,
                   mostrar_progresiva=False,
                   mostrar_regresiva=False,
                   mostrar_exacta=False,
                   mostrar_error=False):
    """
    Genera e imprime una tabla de derivadas para una función continua.

    Parámetros
    ----------
    f              : función objetivo f(x)
    x_vals         : lista o array de puntos donde evaluar
    h              : tamaño del paso para las diferencias finitas
    exact_df       : función analítica de la primera derivada (requerida para
                     mostrar_exacta y mostrar_error)
    titulo         : encabezado que se imprime sobre la tabla
    mostrar_fx     : columna F(x)              — siempre True por defecto
    mostrar_central    : columna F'(x) central
    mostrar_segunda    : columna F''(x) central
    mostrar_progresiva : columna F'(x) progresiva O(h²)
    mostrar_regresiva  : columna F'(x) regresiva  O(h²)
    mostrar_exacta     : columna F'(x) exacta analítica
    mostrar_error      : columna |Error| = |F'exacta - F'central|
                         (requiere exact_df y mostrar_central=True)
    """
    datos = {'x': list(x_vals)}

    if mostrar_fx:
        datos['F(x)'] = [f(x) for x in x_vals]

    if mostrar_central:
        datos["F'(x) central"] = [primera_derivada_central(f, x, h) for x in x_vals]

    if mostrar_segunda:
        datos["F''(x) central"] = [segunda_derivada_central(f, x, h) for x in x_vals]

    if mostrar_progresiva:
        datos["F'(x) prog"] = [primera_derivada_progresiva_o2(f, x, h) for x in x_vals]

    if mostrar_regresiva:
        datos["F'(x) reg"] = [primera_derivada_regresiva_o2(f, x, h) for x in x_vals]

    if mostrar_exacta:
        if exact_df is None:
            raise ValueError("Necesitás pasar 'exact_df' para mostrar la derivada exacta.")
        datos["F'(x) exacta"] = [exact_df(x) for x in x_vals]

    if mostrar_error:
        if exact_df is None or not mostrar_central:
            raise ValueError(
                "Para mostrar el error necesitás pasar 'exact_df' "
                "y activar 'mostrar_central=True'."
            )
        datos["|Error|"] = [
            abs(exact_df(x) - primera_derivada_central(f, x, h)) for x in x_vals
        ]

    df = pd.DataFrame(datos)

    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"  h = {h}")
    print(f"{'='*60}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.8f}"))
    print(f"{'='*60}\n")

    return df


# =========================================================
# MÓDULO 2: DATOS DISCRETOS (TABLAS - Ejercicios 6 y 7)
# =========================================================

def tabla_discreta(t, x, titulo="Tabla de Datos Discretos"):
    """
    Genera e imprime una tabla con posición, velocidad y aceleración
    a partir de datos discretos.

    Parámetros
    ----------
    t      : array de tiempos (o variable independiente)
    x      : array de posiciones (o variable dependiente)
    titulo : encabezado que se imprime sobre la tabla

    Retorna
    -------
    DataFrame con columnas: t, x, V (velocidad), A (aceleración)
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(t)

    if len(x) != n:
        raise ValueError("Los arreglos 't' y 'x' deben tener el mismo largo.")

    v = np.zeros(n)
    a = np.zeros(n)

    # --- Velocidad ---
    for i in range(n):
        if i == 0:
            v[i] = (x[i+1] - x[i]) / (t[i+1] - t[i])          # Progresiva
        elif i == n - 1:
            v[i] = (x[i] - x[i-1]) / (t[i] - t[i-1])           # Regresiva
        else:
            v[i] = (x[i+1] - x[i-1]) / (t[i+1] - t[i-1])       # Central

    # --- Aceleración (derivada de v) ---
    for i in range(n):
        if i == 0:
            a[i] = (v[i+1] - v[i]) / (t[i+1] - t[i])
        elif i == n - 1:
            a[i] = (v[i] - v[i-1]) / (t[i] - t[i-1])
        else:
            a[i] = (v[i+1] - v[i-1]) / (t[i+1] - t[i-1])

    df = pd.DataFrame({
        't':  t,
        'x':  x,
        'V':  v,
        'A':  a,
    })

    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"{'='*60}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"{'='*60}\n")

    return df


# =========================================================
# EJEMPLO DE USO
# =========================================================
if __name__ == '__main__':

    # ---- Módulo 1: función continua ----
    def f(x):      return math.exp(x) * math.sin(x)
    def df_ex(x):  return math.exp(x) * (math.sin(x) + math.cos(x))

    puntos = [0.9, 1.0, 1.1]
    h = 0.01

    tabla_continua(
        f, puntos, h,
        exact_df       = df_ex,
        titulo         = "f(x) = e^x · sin(x)",
        mostrar_fx         = True,
        mostrar_central    = True,
        mostrar_segunda    = True,
        mostrar_progresiva = True,
        mostrar_regresiva  = True,
        mostrar_exacta     = True,
        mostrar_error      = True,
    )

    # ---- Módulo 2: datos discretos ----
    tiempos    = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    posiciones = [0, 1.2, 2.1, 2.8, 3.2, 3.4, 3.5]

    tabla_discreta(tiempos, posiciones, titulo="Movimiento — datos discretos")