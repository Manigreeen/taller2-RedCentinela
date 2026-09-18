import math
import random

from optimization.problem import SmartGridOptimizationProblem
from optimization.result import Configuration, OptimizationResult


def configuration_score(
    problem: SmartGridOptimizationProblem, configuration: Configuration
) -> float:
    """
    Combina cobertura, redundancia y exposición en un puntaje a maximizar.

    Tips:
    - Use problem.score_components(configuration); ya retorna cobertura,
      redundancia y exposición en ese orden.
    """
    coverage, redundancy, exposure = problem.score_components(configuration)
    return coverage - redundancy - exposure


def hill_climbing(problem: SmartGridOptimizationProblem, initial_configuration: Configuration, max_iterations: int = 500,) -> OptimizationResult:
    """
    Ejecuta ascenso de colina con mejora estricta.

    Debe examinar todos los vecinos, seleccionar el de mayor puntaje y
    conservar el orden entregado por el problema para desempatar. La búsqueda
    termina cuando no existe una mejora estricta o se alcanza el límite.

    Tips:
    - problem.neighbors(current) retorna vecinos válidos en el orden que debe
      usarse para desempatar.
    - Cada llamada a configuration_score(...) cuenta como una evaluación.
    - Inicialice los historiales con la configuración inicial y agregue solo las
      mejoras aceptadas antes de retornar el OptimizationResult.
    """
    
    estado_actual =initial_configuration
    puntaje_actual = configuration_score (problem, estado_actual)
    
    evaluaciones = 1 
    iteraciones= 0
    
    historial = [estado_actual]
    historial_puntajes = [puntaje_actual]
    
    mejora_el_estado=True
    while iteraciones < max_iterations and mejora_el_estado:
        vecinos = problem.neighbors(estado_actual)
        if vecinos:
            mejor_vecino= None
            puntaje_mejor_vecino = float("-inf")
            
            for vecino in vecinos:
                puntaje_vecino = configuration_score(problem, vecino)
                evaluaciones += 1
                if puntaje_vecino > puntaje_mejor_vecino:
                    mejor_vecino = vecino
                    puntaje_mejor_vecino = puntaje_vecino
            
            if puntaje_mejor_vecino > puntaje_actual:
                estado_actual = mejor_vecino
                puntaje_actual = puntaje_mejor_vecino
                iteraciones += 1
                historial.append(estado_actual)
                historial_puntajes.append(puntaje_actual)
            else:
                mejora_el_estado = False
                
        else:
            mejora_el_estado = False
    
    return OptimizationResult(best_configuration = estado_actual, best_score = puntaje_actual, evaluations = evaluaciones, iterations = iteraciones, history= historial, score_history = historial_puntajes)


def cooling_schedule(initial_temperature: float, cooling_rate: float, iteration: int) -> float:
    """
    Retorna el programa geométrico T(t) = T0 * alpha**t.

    Esta función se invoca desde simulated_annealing en cada iteración.
    """
    temperatura= initial_temperature * (cooling_rate **iteration)
    
    return temperatura

def simulated_annealing(problem: SmartGridOptimizationProblem, initial_configuration: Configuration, initial_temperature: float = 20.0, cooling_rate: float = 0.97, max_iterations: int = 500, rng: random.Random | None = None,) -> OptimizationResult:
    """
    Ejecuta recocido simulado para un problema de maximización.

    Debe proponer un vecino aleatorio por iteración, aceptar siempre las
    mejoras y aplicar exp(delta / temperature) en los demás casos. El estado
    actual y el mejor estado encontrado deben conservarse por separado.

    Tips:
    - Seleccione el candidato con rng.choice(problem.neighbors(current)) y use
      exclusivamente rng para conservar la reproducibilidad.
    - Obtenga la temperatura con cooling_schedule(...) y calcule la aceptación
      con delta = puntaje_candidato - puntaje_actual y math.exp(...).
    - Mantenga separados el estado actual y el mejor encontrado; registre el
      estado actual después de cada intento, incluso si se rechaza.
    - Detenga la ejecución cuando la temperatura alcance minimum_temperature.
    """
    rng = rng or random.Random()
    minimum_temperature = 1e-9
    
    estado_actual = initial_configuration
    puntaje_actual =configuration_score(problem, estado_actual)
    mejor_estado= estado_actual
    mejor_puntaje= puntaje_actual
    evaluaciones = 1 
    iteraciones= 0 
    
    historial =[estado_actual]
    historial_puntajes = [puntaje_actual]
    #recordadr que la temperatura varia tras cda iteracion 
    temperatura = cooling_schedule(initial_temperature, cooling_rate, iteraciones)
    hay_vecinos= True
    while iteraciones < max_iterations and temperatura > minimum_temperature and hay_vecinos:
        vecinos = problem.neighbors(estado_actual)
        if len(vecinos) > 0:
            candidato = rng.choice(vecinos)
            puntaje_candidato = configuration_score(problem, candidato)
            evaluaciones +=1
            diferencia = puntaje_candidato - puntaje_actual
            aceptar_candidato = False
            if diferencia > 0:
                aceptar_candidato = True
            else:
                probabilidad_aceptacion = math.exp(diferencia / temperatura)
                numero_aleatorio = rng.random()
                if numero_aleatorio < probabilidad_aceptacion:
                    aceptar_candidato = True
            if aceptar_candidato == True:
                estado_actual= candidato
                puntaje_actual = puntaje_candidato
                if puntaje_actual > mejor_puntaje:
                    mejor_estado = estado_actual
                    mejor_puntaje = puntaje_actual
            iteraciones +=1
            historial.append(estado_actual)
            historial_puntajes.append(puntaje_actual)
                
            temperatura = cooling_schedule(initial_temperature, cooling_rate, iteraciones)

        else:
            hay_vecinos = False
            
    return OptimizationResult(best_configuration=mejor_estado, best_score=mejor_puntaje, evaluations=evaluaciones, iterations=iteraciones, history=historial, score_history=historial_puntajes)


def one_point_crossover(
    parent1: Configuration, parent2: Configuration, rng: random.Random
) -> tuple[Configuration, Configuration]:
    """
    Realiza un cruce de un punto y retorna dos descendientes.

    La reparación de la cantidad de módulos se realiza posteriormente.

    Tips:
    - Seleccione con rng un corte interior, entre las posiciones 1 y len-1.
    - Cada descendiente combina el prefijo de un padre con el sufijo del otro.
    - Retorne tuplas y no repare aquí los descendientes.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud")
    if len(parent1) < 2:
        return parent1, parent2

    punto = rng.randint(1, len(parent1) - 1)
    hijo1 = parent1[:punto] + parent2[punto:]
    hijo2 = parent2[:punto] + parent1[punto:]
    
    return hijo1, hijo2


def swap_mutation(
    individual: Configuration, mutation_probability: float, rng: random.Random
) -> Configuration:
    """
    Aplica mutación por intercambio con la probabilidad indicada.

    Cuando ocurre una mutación, intercambia un bit activo y uno inactivo para
    conservar la cantidad de módulos instalados.

    Tips:
    - Use rng.random() para decidir si se aplica la mutación.
    - Identifique por separado los índices activos e inactivos y seleccione uno
      de cada grupo con rng.choice(...).
    - Si alguno de los dos grupos está vacío, no hay un intercambio posible.
    - Retorne una tupla nueva; no modifique el individuo recibido.
    """
    if rng.random() >= mutation_probability:
        return individual
    
    activo = [i for i, bit in enumerate(individual) if bit]
    inactivo = [i for i, bit in enumerate(individual) if not bit]
    
    if not activo or not inactivo:
        return individual
    
    prendido = rng.choice(activo)
    apagado = rng.choice(inactivo)
    
    return tuple(1 if i == apagado else 0 if i == prendido else bit for i, bit in enumerate(individual))

def genetic_algorithm(
    problem: SmartGridOptimizationProblem,
    population_size: int = 40,
    generations: int = 100,
    mutation_probability: float = 0.05,
    elite_size: int = 2,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta un algoritmo genético generacional.

    Debe integrar la población inicial, la selección por torneo, el cruce, la
    reparación, la mutación y el elitismo entregados por el proyecto. Retorna
    el mejor individuo encontrado durante toda la ejecución.

    Tips:
    - Use problem.initial_population(...), problem.tournament_select(...) y
      problem.repair_configuration(...) para las operaciones ya entregadas.
    - Aplique one_point_crossover(...) antes de reparar y swap_mutation(...)
      después de la reparación.
    - Conserve los mejores individuos por elitismo y registre en los historiales
      el mejor global de cada generación.
    """
    rng = rng or random.Random()
    if population_size < 2:
        raise ValueError("La población debe tener al menos dos individuos")
    if generations < 0:
        raise ValueError("El número de generaciones no puede ser negativo")
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("La probabilidad de mutación debe estar entre 0 y 1")
    if not 0 <= elite_size <= population_size:
        raise ValueError("elite_size debe estar entre 0 y population_size")

    poblacion = problem.initial_population(population_size, rng)
    evaluaciones = 0
    mejor_config = None
    mejor_puntaje = float("-inf")
    historial = []
    historial_puntaje = []

    for _ in range(generations):
        puntajes = [configuration_score(problem, c) for c in poblacion]
        evaluaciones += len(poblacion)

        for config, puntaje in zip(poblacion, puntajes):
            if puntaje > mejor_puntaje:
                mejor_config = config
                mejor_puntaje = puntaje

        historial.append(mejor_config)
        historial_puntaje.append(mejor_puntaje)

        top = sorted(zip(poblacion, puntajes), key=lambda pair: pair[1], reverse=True)
        nueva_poblacion = [config for config, _ in top[:elite_size]]

        while len(nueva_poblacion) < population_size:
            p1 = problem.tournament_select(poblacion, puntajes, rng)
            p2 = problem.tournament_select(poblacion, puntajes, rng)
            c1, c2 = one_point_crossover(p1, p2, rng)
            c1 = problem.repair_configuration(c1, rng)
            c2 = problem.repair_configuration(c2, rng)
            c1 = swap_mutation(c1, mutation_probability, rng)
            c2 = swap_mutation(c2, mutation_probability, rng)
            nueva_poblacion.append(c1)
            if len(nueva_poblacion) < population_size:
                nueva_poblacion.append(c2)

        poblacion = nueva_poblacion

    return OptimizationResult(
        best_configuration=mejor_config,
        best_score=mejor_puntaje,
        evaluations=evaluaciones,
        iterations=generations,
        history=historial,
        score_history=historial_puntaje,
    )