from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


def get_element(cf_population):
    random_numer = random.uniform(0, 1)
    for j, cf_p in enumerate(cf_population):
        if random_numer <= cf_p[0]:
            return cf_p[1]


def roulette_wheel_selection(items, knapsack_max_capacity, populatopn, population_size, elite_size):
    relative_fitness_population = []
    total_fitness = sum(fitness(items, knapsack_max_capacity, p) for p in population)
    for p in population:
        relative_fitness_population.append(
            (fitness(items, knapsack_max_capacity, p) / total_fitness, p))  # Relative fitness
    relative_fitness_population.sort()
    relative_fitness_population.reverse()

    cumulative_fitness_population = [relative_fitness_population[0]]
    for i in range(1, len(relative_fitness_population)):
        cumulative_fitness_population.append(
            (relative_fitness_population[i][0] + cumulative_fitness_population[i - 1][0],
             relative_fitness_population[i][1]))  # Cumulative fitness
    return cumulative_fitness_population


def crossover(p_size, cf_population):
    gen = []
    for _ in range(int(p_size/2)):
        element1 = get_element(cf_population)
        element2 = get_element(cf_population)

        el1_1, el1_2 = element1[:int(len(element1) / 2)], element1[int(len(element1) / 2):]
        el2_1, el2_2 = element2[:int(len(element2) / 2)], element2[int(len(element2) / 2):]

        gen.append(el1_1 + el2_2)
        gen.append(el2_1 + el1_2)
    return gen


def mutation(gen):
    for element in gen:
        rand_int = random.randint(0, len(element)-1)
        element[rand_int] = not element[rand_int]


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
elite_size = 5
generations = 200
n_selection = 20
n_elite = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)
    # TODO: implement genetic algorithm

    # Roulette wheel selection
    cumulative_fitness_population = roulette_wheel_selection(items, knapsack_max_capacity, population, population_size, elite_size)

    # Crossover
    next_gen = crossover(population_size, cumulative_fitness_population)

    # Mutation
    mutation(next_gen)

    population = []

    population.extend([cumulative_fitness_population[i][1] for i in range(elite_size)])
    population.extend(next_gen[:population_size-elite_size])

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
