import sympy as sp

# ==========================================
# 1. DATOS DE ENTRADA (Modificá esta sección)
# ==========================================
# Definimos las variables simbólicas base (no borrar, cambiar acorde al ejercicio)
x = sp.symbols('t')
y = sp.Function('y')

# A. Escribí el lado derecho de tu ecuación diferencial: y' = ...
# IMPORTANTE: Usá sp.exp(), sp.sin(), sp.cos(), etc.
# Ejemplo: y(t) + t**2
expresion_edo = x + y(x)

# B. Condiciones iniciales para la solución particular: y(t0) = y0
# Si tu ejercicio pide solo la solución general, cambiá a False
calcular_particular = True
x0 = 0
y0 = 1


# ==========================================
# 2. PROCESAMIENTO (No tocar a partir de acá)
# ==========================================
print("="*55)
print("     RESOLUTOR SIMBÓLICO DE EDOs (con SymPy)")
print("="*55)

# Armamos la ecuación matemática: y'(t) = expresion
edo = sp.Eq(y(x).diff(x), expresion_edo)

print("\n[1] Ecuación Diferencial a resolver:")
sp.pprint(edo)

try:
    # 1. Intentamos buscar la Solución General
    sol_general = sp.dsolve(edo, y(x))
    print("\n[2] Solución General encontrada:")
    sp.pprint(sol_general)
    
    # 2. Si se pidió, buscamos la Solución Particular
    if calcular_particular:
        sol_particular = sp.dsolve(edo, y(x), ics={y(x0): y0})
        print(f"\n[3] Solución Particular exacta (con y({x0}) = {y0}):")
        sp.pprint(sol_particular)
        
except NotImplementedError:
    print("\n[!] AVISO: SymPy no pudo encontrar una solución analítica cerrada.")
    print("    Esto es normal en ecuaciones como la de Riccati (Ej: t - y^2),")
    print("    donde se requieren métodos numéricos (RK4) para aproximarla.")
except Exception as e:
    print(f"\n[!] Ocurrió un error inesperado al intentar resolver: {e}")