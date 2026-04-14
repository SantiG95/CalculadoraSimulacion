import numpy as np
def monte_carlo_with_ci(f, a, b, N):
    x = np.random.uniform(a, b, N)
    f_x = f(x)
    mean_f = np.mean(f_x)
    s = np.std(f_x, ddof=1) # Usando N-1
    z = 1.96 # Para confianza del 95%
    SE = s / np.sqrt(N)
    margin_error = z * SE
    IC = ((b-a)*mean_f - margin_error, (b-a)*mean_f + margin_error)

    return (b - a) * mean_f, IC

integral, IC = monte_carlo_with_ci(lambda x: np.log(x), 1, 5, 10000)
print(f"Estimación: {integral:.6f}")
print(f"IC 95%: [{IC[0]:.6f}, {IC[1]:.6f}]")