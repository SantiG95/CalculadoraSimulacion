import math
import numpy as np
import matplotlib.pyplot as plt

def h(x):
 return math.cos(x)
def g(x):
 return math.cos(x) + x

def metodo_punto_fijo(x0, tol=1e-5, max_iter=100):
    x = x0
    iter_values = [x0]
    for i in range(max_iter):
        x_new = g(x)
        iter_values.append(x_new)
        if abs(x_new - x) < tol:
            print("Tolerance exceeded...")
            break
        x = x_new
    return x_new, iter_values
x0 = 1.0 # Valor inicial
root, iter_values = metodo_punto_fijo(x0)

print(f"La raíz aproximada es: {root}")

# Graficar la función original y el proceso de iteración
x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 400)
y_vals = np.cos(x_vals)
plt.plot(x_vals, y_vals, label='$f(x) = cos(x)$')
plt.scatter(iter_values, [f(x) for x in iter_values], color='red', zorder=5)
plt.plot(iter_values, [f(x) for x in iter_values], color='red', linestyle='--', zorder=5)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.title('Método del Punto Fijo para $f(x) = cos(x)$')
plt.scatter([root], [0], color='red', zorder=5, label=f'Raíz: {root}')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.show()

print(f"La raíz aproximada es: {root}")