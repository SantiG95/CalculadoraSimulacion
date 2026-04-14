import random
from collections import Counter

# 1. Definimos los 32 equipos y su "fuerza" (puntaje arbitrario o ranking ELO)
equipos = {
    "Argentina": 90, "Francia": 89, "Brasil": 88, "Inglaterra": 86,
    "España": 85, "Alemania": 83, "Portugal": 84, "Países Bajos": 82,
    "Italia": 81, "Uruguay": 80, "Croacia": 79, "Bélgica": 78,
    "Marruecos": 77, "Colombia": 76, "Senegal": 75, "Japón": 74,
    "Estados Unidos": 73, "México": 72, "Suiza": 71, "Ecuador": 70,
    "Dinamarca": 69, "Suecia": 68, "Corea del Sur": 67, "Gales": 66,
    "Polonia": 65, "Serbia": 64, "Irán": 63, "Arabia Saudita": 62,
    "Australia": 61, "Túnez": 60, "Costa Rica": 59, "Canadá": 58
}

def simular_partido(equipo1, equipo2):
    """
    Simula un partido entre dos equipos usando sus fuerzas.
    El equipo con más fuerza tiene más probabilidades, pero no la victoria asegurada.
    """
    fuerza1 = equipos[equipo1]
    fuerza2 = equipos[equipo2]
    
    # Calculamos la probabilidad de que gane el equipo 1
    prob_ganar1 = fuerza1 / (fuerza1 + fuerza2)
    
    # Usamos un número aleatorio para decidir el ganador
    if random.random() < prob_ganar1:
        return equipo1
    else:
        return equipo2

def simular_mundial(lista_equipos):
    """Simula un mundial completo desde 16avos hasta la final y devuelve al campeón."""
    ronda_actual = lista_equipos.copy()
    
    # Mezclamos aleatoriamente para armar las llaves iniciales (sorteo)
    random.shuffle(ronda_actual)
    
    # Mientras haya más de 1 equipo, seguimos jugando rondas
    while len(ronda_actual) > 1:
        siguiente_ronda = []
        # Jugamos los partidos emparejando de a 2
        for i in range(0, len(ronda_actual), 2):
            ganador = simular_partido(ronda_actual[i], ronda_actual[i+1])
            siguiente_ronda.append(ganador)
        ronda_actual = siguiente_ronda
        
    return ronda_actual[0] # El único equipo que queda es el campeón

def montecarlo_mundial(n_simulaciones):
    """Ejecuta N simulaciones del mundial y cuenta las victorias."""
    lista_equipos = list(equipos.keys())
    campeones = Counter()
    
    # Bucle de Montecarlo: repetimos el evento complejo miles de veces
    for _ in range(n_simulaciones):
        campeon = simular_mundial(lista_equipos)
        campeones[campeon] += 1
        
    # Calculamos los porcentajes finales
    resultados = []
    for equipo, victorias in campeones.most_common():
        probabilidad = (victorias / n_simulaciones) * 100
        resultados.append((equipo, probabilidad))
        
    return resultados

# ==========================================
# Ejecución del Script
# ==========================================
if __name__ == "__main__":
    N_SIMULACIONES = 10000
    print(f"Simulando {N_SIMULACIONES} Mundiales usando el Método Montecarlo...\n")
    
    resultados_finales = montecarlo_mundial(N_SIMULACIONES)
    
    print(f"{'EQUIPO':<18} | {'PROBABILIDAD DE CAMPEONAR'}")
    print("-" * 45)
    
    for equipo, prob in resultados_finales:
        # Mostramos solo los que tienen más de un 1% de chance para limpiar la salida
        if prob > 1.0: 
            print(f"{equipo:<18} | {prob:.2f}%")