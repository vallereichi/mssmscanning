import numpy as np
import random
import typing
import time
from collections.abc import Callable
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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
        point: list[float]
    
    ) -> float:

    return point[0]**2 + point[1]**2
obj_parabolic: Callable[[list[float]], float] = parabolic

def two_valleys(point: list[float]) -> float:
    x = point[0]
    y = point[1]
    valley1 = 5 * np.exp(-((x - 20)**2 + (y - 2)**2)/100)
    valley2 = 10 * np.exp(-((x + 20)**2 + (y + 2)**2)/300)

    return -valley1 - valley2 + 100

def four_valleys(point: list[float]) -> float:
    x = point[0]
    y = point[1]
    valley1 = np.exp(-((x - 20)**2 + (y - 20)**2)/200)
    valley2 = np.exp(-((x + 20)**2 + (y - 20)**2)/200)
    valley3 = np.exp(-((x - 20)**2 + (y + 20)**2)/200)
    valley4 = np.exp(-((x + 20)**2 + (y + 20)**2)/200)
    return -valley1 - valley2 - valley3 - valley4 + 1000




"""
differential evolution
"""
def diver(
        population_size: int,
        ranges: list[list[float]], 
        mutation_scale_factor: float,
        crossover_rate: float,
        objective_function: Callable[[list[float]], float],
        conv_thresh: float = convergence_threshold,
        max_iter: int = MAX_ITERATIONS

    ) -> tuple[list[list[float]], float]:


    # give some feedback on the configuration
    print("population size: ", population_size)
    print("F: ", mutation_scale_factor)
    print("Cr: ", crossover_rate)
    print("convergence threshold: ", conv_thresh)
    print("break after max iterations:", max_iter)

    population_list: list = []
    improvement_list: list = []

    # initialize the population
    population: list = [[random.uniform(ranges[i][0], ranges[i][1]) for i in range(len(ranges))] for _ in range(population_size)]
    population_list.append(population)
    
    improvement: float = 0.
    update_times: list = []    

    # main update loop
    while len(population_list) <= max_iter:
        start_timer = time.time()
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
            improvement = 0
            
        if abs(target_lh) > abs(trial_lh):
            current_population[target_vector_id] = trial_vector
            improvement = abs(trial_lh - target_lh)
            
        end_timer = time.time()

        new_population = current_population.copy()
        population_list.append(new_population)
        improvement_list.append(improvement)
        update_times.append(start_timer - end_timer)

        print("population: ", len(population_list), "       improvement =", improvement)

        
        # break condition
        if 0 < improvement < conv_thresh:
            break

    return population_list, improvement_list, update_times
                









if __name__ == "__main__":
    populations, improvements, update_times = diver(population_size, ranges, mutation_scale_factor, crossover_rate, obj_parabolic)

    # create animation
    if len(ranges) == 2:
    
        metadata = dict(title="rand/1/bin", artist="Valentin Reichenspurner")
        writer = PillowWriter(fps=15, metadata=metadata)

        def parabolic_mesh(x: float, y:float) -> float:
            return x**2 + y**2

        y = list(np.linspace(ranges[1][0], ranges[1][1], 100))
        x = list(np.linspace(ranges[0][0], ranges[0][1], 100))
        X,Y = np.meshgrid(x,y)

        z = parabolic_mesh(X,Y)

        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X,Y,z, cmap='plasma')
        ax1.set_axis_off()


        ax2 = fig.add_subplot(122)
        contour = ax2.contour(X,Y,z, levels=50)
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_xlim(ranges[0][0], ranges[0][1])
        ax2.set_ylim(ranges[1][0], ranges[1][1])
        ax2.set_aspect('equal', adjustable='box')
        fig.colorbar(contour, label="p(x,y)")

        scatter = ax2.scatter([], [], color='magenta', alpha=1, marker='.')

        with writer.saving(fig, 'diver_animation.gif', 100):

            for frame in populations:
                scatter.set_offsets(frame)
                writer.grab_frame()



    


    
    