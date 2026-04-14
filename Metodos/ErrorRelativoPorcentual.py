def error_relativo_porcentual(valor_real, valor_aprox):
    """
    Calcula el error relativo porcentual entre el valor analítico exacto
    y el valor aproximado por un método numérico.
    """
    if valor_real == 0:
        return float('inf') # Para evitar división por cero si el valor real es exactamente 0
        
    error = abs((valor_real - valor_aprox) / valor_real) * 100
    return error