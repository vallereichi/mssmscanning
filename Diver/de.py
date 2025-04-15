"""
Differential Evolution.
An optimizing algorithm with fast convergence and good stability for high dimensional parameter spaces.
"""

from collections.abc import Callable
import time
import random
import numpy as np
import vallog as vl

import objectives

# type hinting
type Vector = list[float]
type Population = list[Vector]
type Likelihood = Callable[[Vector], float]

# initialize logs
msg = vl.Logger("Release")


# function declarations
def get_best_vector(population: Population, objective_function: Likelihood) -> tuple[Vector, int, float]:
    """find the best vector in a population"""
    best_vector: Vector = population[0]
    best_vector_id: int = 0
    best_lh: float = 0.0

    for i, point in enumerate(population):
        lh: float = objective_function(point)
        if abs(lh) < best_lh or best_lh == 0.0:
            best_vector = point
            best_vector_id = i
            best_lh = lh

    return best_vector, best_vector_id, best_lh


def select_target_ramdom(population: Population) -> tuple[Vector, int]:
    """select a random target vector from the population"""
    target_vector_id = random.randint(0, len(population) - 1)
    target_vector: list[float] = population[target_vector_id]
    return target_vector, target_vector_id


def select_target_best(population: Population, objective_function: Likelihood) -> tuple[Vector, int]:
    """select the best vector from a population as the target vector"""
    best_vector, best_vector_id, _ = get_best_vector(population, objective_function)
    return best_vector, best_vector_id


def mutation_simple(population: Population, tagret_vector_id: int, mutation_scale_factor: float = 0.8) -> Vector:
    """create a donor vector from 3 randomly selected points of the population"""
    population_cpy = population.copy()
    population_cpy.pop(tagret_vector_id)
    a, b, c = random.sample(population_cpy, 3)
    donor_vector: Vector = [a[i] + mutation_scale_factor * (b[i] - c[i]) for i in range(len(a))]
    return donor_vector


def crossover(target_vector: Vector, donor_vector: Vector, crossover_rate: float = 0.8) -> Vector:
    """create the trial vector from the selected target and the donor vector"""
    trial_vector: list[float] = []
    for i, _ in enumerate(target_vector):
        random_value = random.uniform(0, 1)
        if random_value <= crossover_rate:
            trial_vector.append(donor_vector[i])
        else:
            trial_vector.append(target_vector[i])

    random_dimension = random.randint(0, len(target_vector) - 1)
    trial_vector[random_dimension] = donor_vector[random_dimension]
    return trial_vector


def selection(
    target_vector: Vector,
    target_vector_id: int,
    trial_vector: Vector,
    population: Population,
    objective_function: Likelihood,
) -> tuple[Population, float]:
    """Select either the target or the trial vector for the next generation"""
    target_lh = objective_function(target_vector)
    trial_lh = objective_function(trial_vector)
    improvement: float = 0.0

    if abs(target_lh) < abs(trial_lh):
        population[target_vector_id] = target_vector

    if abs(target_lh) > abs(trial_lh):
        population[target_vector_id] = trial_vector
        improvement = abs(trial_lh - target_lh)

    return population, improvement


# run differential evolution
def diver(
    population_size: int,
    ranges: list[list[float]],
    mutation_scale_factor: float,
    crossover_rate: float,
    objective_function: Likelihood,
    conv_thresh: float = 1e-3,
    max_iter: int = 10e3,
) -> tuple[Population, list[float], list[float]]:
    """
    running the differential evolution algorithm with user specific configuration

    *parameters*\n
    population_size:        determines how many points are initialized for the DE run

    ranges:                 specifies the lower and upper bounds of the
                            parameterspace

    mutation_scale_factor:  the factor by which the difference vector will be
                            scaled in the mutation step. Larger values lead to higher diversity

    crossover_rate:         used in the crossover step to create the trial
                            vector. larger values are suited for higher diversity in the population while small values are effective for uncorrelated dimensions

    cov_thresh:             Once the improvement of a generation reaches
                            this threshold, the algorithm will stop and output the result

    max_iter:               maximum iterations the algorithm will perform
                            before it stops and outputs the result
    """

    # give some feedback on the configuration
    msg.sep()
    msg.heading("starting differential evolution with the following configuration")
    msg.log(f"NP: \t\t{population_size}", vl.info)
    msg.log(f"F: \t\t{mutation_scale_factor}", vl.info)
    msg.log(f"Cr: \t\t{crossover_rate}", vl.info)
    msg.log(f"conv_thresh: \t{conv_thresh}", vl.info)
    msg.log(f"max_iter: \t{max_iter}", vl.info)
    msg.sep(" ")

    population_list: list = []
    improvement_list: list = []

    # initialize the population
    population: list = [
        [random.uniform(ranges[i][0], ranges[i][1]) for i in range(len(ranges))] for _ in range(population_size)
    ]
    population_list.append(population)

    improvement: float = 0.0
    update_times: list = []

    # main update loop
    while len(population_list) <= max_iter:
        with vl.Timer() as timer:
            current_population = population_list[-1]

            # select a target vector
            target_vector, target_vector_id = select_target_ramdom(current_population)
            msg.log(f"target vector: {target_vector}", vl.debug)

            # mutation
            donor_vector = mutation_simple(current_population, target_vector_id, mutation_scale_factor)
            msg.log(f"donor vector: {donor_vector}", vl.debug)

            # crossover
            trial_vector = crossover(target_vector, donor_vector, crossover_rate)
            msg.log(f"trial vector: {trial_vector}", vl.debug)

            # selection
            current_population, improvement = selection(
                target_vector, target_vector_id, trial_vector, current_population, objective_function
            )

        # msg.log(f"Timer exceptions: {timer.}")

        new_population = current_population.copy()
        population_list.append(new_population)
        improvement_list.append(improvement)
        update_times.append(timer.elapsed_time)

        msg.log(f"population: {len(population_list)}\t\timprovement = {improvement}", vl.debug)

        # break condition
        if 0 < improvement < conv_thresh:
            break

    # information for the user
    msg.heading("Differential Evolution has finished with")
    msg.log(f"{len(population_list)} generations", vl.info)
    msg.log(f"best final vector: {get_best_vector(population_list[-1], objective_function)[0]}", vl.info)
    msg.sep()

    return population_list, improvement_list, update_times


if __name__ == "__main__":
    """global constants"""
    NP: int = 10
    F: float = 0.5
    Cr: float = 0.5
    parameter_space: list[tuple[float]] = [(-50, 50), (-50, 50)]

    objective = objectives.gaussian

    convergence_threshold: float = 1e-5
    MAX_ITERATIONS: int = 10000

    populations, improvements, times = diver(NP, parameter_space, F, Cr, objective)
