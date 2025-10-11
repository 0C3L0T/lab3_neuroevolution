
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import random
from typing import List

from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding


import controllers
from individual import Individual, init_individual, load_individual, mutate_crossover_individuals, store_individual
from simulation import train_individual
from status import Status


type Population = List[Individual]

NUM_BRAIN_ACTORS = os.cpu_count() // 2
BRAIN_POPULATION_SIZE = 24
ARENA_SIZE = 5
BODY_POPULATION_SIZE = 6  # should be multiple of 6

def init_population(population_size: int, _init_individual) -> Population:
    return [_init_individual() for _ in range(population_size)]

def load_population(status: Status) -> Population | None:
    """
    load population from a location in status object or none if not available
    """

    checkpoint_root = Path(status.checkpoint_dir)
    if not checkpoint_root.exists() or not checkpoint_root.is_dir():
        return None

    gens = [
        d for d in checkpoint_root.iterdir()
        if d.is_dir() and d.name.startswith("generation_")
    ]
    if not gens:
        return None

    latest_gen_dir = max(gens, key=lambda p: int(p.name.split("_")[-1]))
    print(f"loading population from {latest_gen_dir}")

    population: Population = []
    for file_path in latest_gen_dir.iterdir():
        if file_path.is_file():
            ind = load_individual(file_path)
            ind.fitness = None
            population.append(ind)

    return population or None

def store_generation(status: Status, population: Population, generation: int) -> None:
    """
    store generation population on disk
    """
    checkpoint_dir = Path(f"{status.checkpoint_dir}/generation_{generation}")

    # create if doesn't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for individual in population:
        store_individual(dir_path=checkpoint_dir, individual=individual)


def train_individual_wrapper(individual: Individual) -> Individual:
    # EvoTorch inits the Ray clusters

    print('train individual wrapper')

    train_individual(individual, BRAIN_POPULATION_SIZE, NUM_BRAIN_ACTORS)

    return individual

def train_population(population: Population, max_workers: int) -> Population:

    print('train population')

    to_train = [ind for ind in population if ind.fitness is None]
    print(f'skipping training for {len(population) - len(to_train)} individuals.')

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        _ = list(executor.map(train_individual_wrapper, to_train))

    # put back the trained ids
    trained_ids = {id(ind): ind for ind in to_train}
    for i, ind in enumerate(population):
        if id(ind) in trained_ids:
            population[i] = trained_ids[id(ind)]

    return population


def evolve_population(population: Population, _init_individual):
    '''
    replace bottom halve with evolved children
    copy all the indivuals to the next generation
    '''

    # select best half of population
    parents = tournament_selection(population, BODY_POPULATION_SIZE // 2, ARENA_SIZE)

    # crossover mutate
    parents_copy = deepcopy(parents)
    child_genomes = mutate_crossover_population(parents_copy)


    children = [
        _init_individual(genome=genome) for genome in child_genomes
    ]


    return parents + children

def tournament_selection(
        population: Population,
        nr_parents: int,
        arena_size: int
        ) -> Population:
    '''
    select nr_parents from population through tournament
    return Population of size nr_parents
    '''
    selected = []

    for _ in range(nr_parents):
        # take random individuals
        arena = random.sample(population, arena_size)

        # sort by fitness (fittest first)
        arena.sort(key=lambda ind: ind.fitness, reverse=True)

        # Add best to children
        selected.append(arena[0])

    return selected

def mutate_crossover_population(population: Population) -> Population:
    '''
    '''
    assert(len(population) % 3 == 0)

    children: List[Individual] = []

    child_genomes = []
    for i in range(0, len(population), 3):
        genomes = mutate_crossover_individuals(population[i:i+3])
        child_genomes.extend(genomes)

    return child_genomes