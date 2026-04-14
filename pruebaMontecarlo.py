from Metodos.MetodoMonteCarlo import metodoMonteCarloIntegral

f = lambda x: x**2
res = metodoMonteCarloIntegral(f, (0, 1), 10000, actual_value=1/3, seed=123)
print(res)