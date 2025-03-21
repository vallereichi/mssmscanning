import numpy as np
import random
import typing
from collections.abc import Callable
import matplotlib.pyplot as plt

"""
global constants
"""
population_size: int = 10
mutation_scale_factor: float = 0.5
crossover_rate: float = 0.5
ranges: list[tuple[float]] = [(-50,50), (-50,50)]

convergence_threshold: float = 1e-5
MAX_ITERATIONS: int = 10000


"""
objective functions
"""
def gaussian(
        point: list[float],
        mu: float = 50,
        sigma: float = 30

    ) -> float:
    
    return np.sum([-0.5 * np.log(2*np.pi*sigma**2) - (1/(2*sigma**2)) * (x - mu)**2 for x in point])
obj_gaussian: Callable[[list[float], float, float], float] = gaussian


def parabolic(
        point: list[float],
    
    ) -> float:

    if len(point) != 2:
        raise Exception("this objective function can only take two-dimensional inputs")

    return point[0]**2 + point[1]**2
obj_parabolic: Callable[[list[float]], float] = parabolic



"""
differential evolution
"""
def diver(
        population_size: int,
        ranges: list[list[float]], 
        mutation_scale_factor: float,
        crossover_rate: float,
        objective_function: Callable[[list[float]], float]

    ) -> tuple[list[list[float]], float]:

    population_list: list = []
    improvement_list: list = []

    # initialize the population
    population: list = [[random.uniform(ranges[i][0], ranges[i][1]) for i in range(len(ranges))] for _ in range(population_size)]
    population_list.append(population)

    
    improvement: float = 0.

    # main update loop
    while len(population_list) <= MAX_ITERATIONS:
        current_population = population_list[-1]

        # select a target vector
        target_vector_id = random.randint(0, population_size-1)
        target_vector: list[float] = current_population[target_vector_id]

        # mutation
        a, b, c = random.sample(current_population, 3)
        donor_vector: list[float] = [a[i] + mutation_scale_factor * (b[i] - c[i]) for i in range(len(a))]

        # crossover
        trial_vector: list[float] = []
        for i in range(len(target_vector)):
            random_value = random.uniform(0,1)
            if random_value <= crossover_rate:
                trial_vector.append(donor_vector[i])
            else:
                trial_vector.append(target_vector[i])

        random_dimension = random.randint(0, len(ranges)-1)
        trial_vector[random_dimension] = donor_vector[random_dimension]

        # selection
        target_lh = objective_function(target_vector)
        trial_lh = objective_function(trial_vector)

        if abs(target_lh) < abs(trial_lh):
            current_population[target_vector_id] = target_vector
        if abs(target_lh) > abs(trial_lh):
            current_population[target_vector_id] = trial_vector

        new_population = current_population.copy()

        population_list.append(new_population)
        improvement = abs(trial_lh - target_lh)
        improvement_list.append(improvement)

        print("current population:", len(population_list), "   improvement =", improvement)

        
        # break condition
        if 0 < improvement < convergence_threshold:
            break

    return population_list, improvement_list
                









if __name__ == "__main__":
    populations, improvements = diver(population_size, ranges, mutation_scale_factor, crossover_rate, obj_gaussian)

    # create animation
    if len(ranges) != 2:
        raise Exception("can only create animation for two dimensional objective functions")

    y = list(np.linspace(ranges[1][0], ranges[1][1], 10000))
    x = list(np.linspace(ranges[0][0], ranges[0][1], 10000))
    z = [parabolic(point) for point in list(zip(x,y))] 
    


    
    