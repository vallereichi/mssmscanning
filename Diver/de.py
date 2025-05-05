"""
Differential Evolution.
An optimizing algorithm with fast convergence and good stability for high dimensional parameter spaces.
"""

from collections.abc import Callable
import inspect
import random
import copy
import vallog as vl
import objectives


# type hinting
type Vector = list[float]
type Population = list[Vector]
type Likelihood = Callable[[Vector], float]
type Space = list[list[float]]

# initialize logs
msg = vl.Logger("Release")


# create a Diver class
class Diver:
    """
    Class containing the specific configuration and the option to run the differential evolution algorithm
    """

    def __init__(
        self,
        parameter_space: Space | None = None,
        objective_function: Likelihood | None = None,
        population_size: int | None = None,
        conv_thresh: float | None = None,
        max_iter: int | None = None,
        debug: bool = False,
        cout: bool = True,
    ) -> None:
        """initialise Diver"""
        # initialise logging
        self.debug = debug
        self.cout = cout
        self.msg = vl.Logger("Debug", cout=self.cout) if debug else vl.Logger("Release", cout=self.cout)

        # set the configuration
        self.parameter_space: Space = [[0, 100], [0, 100]] if parameter_space is None else parameter_space
        self.population_size: int = 10 * len(self.parameter_space) if population_size is None else population_size
        self.dimensions: int = len(self.parameter_space)
        self.conv_thresh: float = 1e-3 if conv_thresh is None else conv_thresh
        self.max_iter: int = 1000 if max_iter is None else max_iter
        self.objective_func: Likelihood = objectives.gaussian if objective_function is None else objective_function

        # prepare the output and create the first generation
        self.population_list: list[Population] = []
        self.current_population: Population = self.initialise_population()
        self.improvements: list[float] = []
        self.update_times: list[float] = []
        self.select_target: Callable[[], tuple[Vector, int]] = self.select_target_random
        self.mutation_scheme: Callable[[], Vector] = self.mutation_simple
        self.mutation_signature: dict = []
        self.crossover_rate: float = 0.8

    def __repr__(self) -> str:
        """print out the configuration"""
        self.msg.sep()
        self.msg.heading("Diver configuration")
        self.msg.log(f"Population size: {self.population_size}", vl.info)
        self.msg.log(f"Parameter Space: {self.dimensions}-dimensional", vl.info)
        self.msg.log(f"Objective function: {self.objective_func.__name__}", vl.info)
        self.msg.log(f"Convergence threshold: {self.conv_thresh}", vl.info)
        self.msg.log(f"Max Iterations: {self.max_iter}", vl.info)
        return ""

    def initialise_population(self) -> Population:
        """create the first population with random points in the parameter space"""
        if self.cout:
            print(self)
        new_population = [
            [
                random.uniform(self.parameter_space[i][0], self.parameter_space[i][1])
                for i in range(len(self.parameter_space))
            ]
            for _ in range(self.population_size)
        ]
        self.population_list.append(copy.deepcopy(new_population))
        return new_population

    def get_best_vector(self, population: Population | None = None) -> tuple[Vector, int, float]:
        """find the best vector in a population"""
        population: Population = self.current_population if population is None else population
        best_vector: Vector = population[0]
        best_vector_id: int = 0
        best_lh: float = 0.0

        for i, point in enumerate(population):
            lh: float = self.objective_func(point)
            if abs(lh) < best_lh or best_lh == 0.0:
                best_vector = point
                best_vector_id = i
                best_lh = lh

        return best_vector, best_vector_id, best_lh

    def select_target_random(self) -> tuple[Vector, int]:
        """select a random target vector from the population"""
        target_vector_id = random.randint(0, len(self.current_population) - 1)
        target_vector: list[float] = self.current_population[target_vector_id]
        return target_vector, target_vector_id

    def select_target_best(self) -> tuple[Vector, int]:
        """select the best vector from a population as the target vector"""
        best_vector, best_vector_id, _ = self.get_best_vector()
        return best_vector, best_vector_id

    def select_target_index(self, index: int) -> tuple[Vector, int]:
        """specify the index of the target vector"""
        target_vector_id = index
        target_vector = self.current_population[target_vector_id]
        return target_vector, target_vector_id

    def mutation_simple(self, target_vector_id: int, mutation_scale_factor: float | None = None) -> Vector:
        """create a donor vector from 3 randomly selected points of the population"""
        F = 0.8 if mutation_scale_factor is None else mutation_scale_factor

        population_cpy = self.current_population.copy()
        population_cpy.pop(target_vector_id)
        a, b, c = random.sample(population_cpy, 3)
        donor_vector: Vector = [a[i] + F * (b[i] - c[i]) for i in range(len(a))]
        return donor_vector

    def crossover(self, target_vector: Vector, donor_vector: Vector, crossover_rate: float) -> Vector:
        """create the trial vector from the selected target and the donor vector"""
        self.crossover_rate = crossover_rate
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
        self,
        target_vector: Vector,
        target_vector_id: int,
        trial_vector: Vector,
    ) -> tuple[Population, float]:
        """Select either the target or the trial vector for the next generation"""

        target_lh = self.objective_func(target_vector)
        trial_lh = self.objective_func(trial_vector)
        improvement: float = 0.0

        if abs(target_lh) < abs(trial_lh):
            self.current_population[target_vector_id] = target_vector

        if abs(target_lh) > abs(trial_lh):
            self.current_population[target_vector_id] = trial_vector
            improvement = abs(trial_lh - target_lh)
            self.msg.log(f"population changed: {target_vector} -> {trial_vector}", vl.debug)

        new_population = self.current_population.copy()

        return new_population, improvement

    def update(
        self,
        crossover_rate: float,
        select_target: Callable[[], tuple[Vector, int]],
        mutation_scheme: Callable[[], Vector],
        **kwargs,
    ) -> tuple[Population, float]:
        """create the next generation"""

        target_vector, target_vector_id = select_target()
        donor_vector = mutation_scheme(target_vector_id, **kwargs)
        trial_vector = self.crossover(target_vector, donor_vector, crossover_rate)
        new_population, improvement = self.selection(target_vector, target_vector_id, trial_vector)

        return new_population, improvement

    def run(
        self,
        crossover_rate: float | None = None,
        select_target: Callable[[], tuple[Vector, int]] | None = None,
        mutation_scheme: Callable[[], Vector] | None = None,
        **kwargs,
    ) -> tuple[list[Population], list[float], list[float]]:
        """start the algorithm"""

        crossover_rate = self.crossover_rate if crossover_rate is None else crossover_rate
        select_target = self.select_target_random if select_target is None else select_target
        self.select_target = select_target
        mutation_scheme = self.mutation_simple if mutation_scheme is None else mutation_scheme
        self.mutation_scheme = mutation_scheme

        mutation_signature = inspect.signature(mutation_scheme)
        mutation_params = (
            [
                (name, param.default)
                for name, param in mutation_signature.parameters.items()
                if param.default is not param.empty
            ]
            if not kwargs
            else kwargs
        )
        self.mutation_signature = mutation_signature

        self.msg.log(f"target: {select_target.__name__}", vl.info)
        self.msg.log(f"mutation scheme: {mutation_scheme.__name__} with {mutation_params}", vl.info)

        while len(self.population_list) < self.max_iter:
            with vl.Timer() as timer:
                new_population, improvement = self.update(crossover_rate, select_target, mutation_scheme, **kwargs)

            self.msg.log(
                f"generation {len(self.population_list)}: took {timer.elapsed_time} seconds and improved by {improvement}",
                vl.debug,
            )

            self.population_list.append(new_population)
            self.improvements.append(improvement)
            self.update_times.append(timer.elapsed_time)

            if 0 < improvement < self.conv_thresh:
                break

        self.msg.heading("Differential Evolution has finished with")
        self.msg.log(f"{len(self.population_list)} generations", vl.info)
        self.msg.log(f"best final vector: {self.get_best_vector()[0]}", vl.info)
        self.msg.sep()
        return self.population_list, self.improvements, self.update_times


if __name__ == "__main__":
    par_space = [[0, 100], [0, 100]]
    objective = objectives.gaussian
    de = Diver(par_space, objective)
    populations, improvements, update_times = de.run()
    print(de.__dict__.keys())
    de.msg.heading("Diver has hinished succesfully")
