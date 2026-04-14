import random
from collections import Counter

# 1. Definimos los datos de los equipos (Fuerza 1-100, Títulos históricos, Efectividad últimos 10 partidos 0.0-1.0)
# Puedes actualizar estos datos con los equipos exactos de los octavos/cuartos de final actuales.
equipos_libertadores = {
    "Flamengo": {"fuerza": 85, "titulos": 3, "racha": 0.80},
    "River Plate": {"fuerza": 84, "titulos": 4, "racha": 0.75},
    "Palmeiras": {"fuerza": 86, "titulos": 3, "racha": 0.85},
    "Boca Juniors": {"fuerza": 81, "titulos": 6, "racha": 0.60},
    "Fluminense": {"fuerza": 82, "titulos": 1, "racha": 0.70},
    "Peñarol": {"fuerza": 76, "titulos": 5, "racha": 0.65},
    "São Paulo": {"fuerza": 80, "titulos": 3, "racha": 0.70},
    "Nacional": {"fuerza": 75, "titulos": 3, "racha": 0.60}
}

def simular_partido(equipo1, equipo2):
    """Simula un partido entre dos equipos usando una distribución normal para añadir aleatoriedad."""
    t1 = equipos_libertadores[equipo1]
    t2 = equipos_libertadores[equipo2]
    
    # Pesos (Puedes ajustar estos multiplicadores para darle más importancia a la historia o a la racha)
    peso_titulos = 2.0  # Cada copa ganada suma 2 puntos de poder
    peso_racha = 15.0   # Tener 100% de efectividad suma 15 puntos
    
    # Calculamos el "Poder Teórico"
    poder_t1 = t1["fuerza"] + (t1["titulos"] * peso_titulos) + (t1["racha"] * peso_racha)
    poder_t2 = t2["fuerza"] + (t2["titulos"] * peso_titulos) + (t2["racha"] * peso_racha)
    
    # Factor Montecarlo: Desviación estándar de 10. 
    # (Un número mayor significa que hay más sorpresas, un número menor hace que casi siempre gane el favorito)
    desviacion = 10 
    rendimiento_t1 = random.gauss(poder_t1, desviacion)
    rendimiento_t2 = random.gauss(poder_t2, desviacion)
    
    # En caso de empate exacto (muy raro en floats), desempatamos por moneda
    if rendimiento_t1 == rendimiento_t2:
        return random.choice([equipo1, equipo2])
        
    return equipo1 if rendimiento_t1 > rendimiento_t2 else equipo2

def simular_fase_eliminatoria(lista_equipos):
    """Simula un cuadro de eliminación directa hasta que quede un campeón."""
    ronda_actual = lista_equipos
    
    while len(ronda_actual) > 1:
        siguiente_ronda = []
        # Emparejamos de a dos (el 0 con el 1, el 2 con el 3, etc.)
        for i in range(0, len(ronda_actual), 2):
            ganador = simular_partido(ronda_actual[i], ronda_actual[i+1])
            siguiente_ronda.append(ganador)
        ronda_actual = siguiente_ronda
        
    return ronda_actual[0]

def montecarlo_libertadores(equipos, iteraciones=10000):
    """Ejecuta la simulación miles de veces y calcula las probabilidades."""
    campeones = []
    nombres_equipos = list(equipos.keys())
    
    print(f"Iniciando simulación de Montecarlo con {iteraciones} iteraciones...\n")
    
    for _ in range(iteraciones):
        # Mezclamos los equipos para simular diferentes cruces en el cuadro (sorteo aleatorio)
        # Si ya tienes la llave definida, puedes eliminar el random.shuffle
        cuadro = nombres_equipos.copy()
        random.shuffle(cuadro) 
        
        campeon = simular_fase_eliminatoria(cuadro)
        campeones.append(campeon)
        
    # Contamos las victorias y calculamos porcentajes
    resultados = Counter(campeones)
    
    print("-" * 40)
    print("🏆 PROBABILIDADES DE GANAR LA LIBERTADORES 🏆")
    print("-" * 40)
    
    for equipo, victorias in resultados.most_common():
        probabilidad = (victorias / iteraciones) * 100
        print(f"{equipo:<15}: {probabilidad:>5.2f}% ({victorias} torneos ganados)")

if __name__ == "__main__":
    # Ejecutamos 10,000 simulaciones para obtener un resultado estadísticamente robusto
    montecarlo_libertadores(equipos_libertadores, iteraciones=10000)