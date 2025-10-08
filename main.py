## Standard library
from concurrent.futures import ProcessPoolExecutor
import os
import random
from typing import List

## Third party libraries
import numpy as np
import torch

import mujoco as mj

## Local libraries
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder, save_graph_as_json

import controllers
from individual import (
    Individual,
    load_individual,
    store_individual,
)

from simulation import show_individual_in_window, train_individual
from status import (
    Status,
    display_training_status,
    load_training_status,
    store_training_status
)

import controllers


# Magic Numbers
NUM_BODY_MODULES = 30  # they've changed this to 30 in the template now
# TODO: make sure to modify this before submitting to AWS
NUM_BRAIN_ACTORS = 2 #os.cpu_count() // 2
NUM_BODY_ACTORS = 2
BODY_POPULATION_SIZE = 8
BRAIN_POPULATION_SIZE = 24
DEFAULT_BODY_ITERATIONS = 100

# --- RANDOM GENERATOR SETUP --- #

#SEED = 42
#RNG = np.random.default_rng(SEED)

GLOBAL_NDE = None
#GLOBAL_HPD = None

# --- DATA SETUP ---
from pathlib import Path
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# --- CUSTOM TYPES --- #
type Population = List[Individual]

def init_nde_hpd():
    global GLOBAL_NDE#, GLOBAL_HPD
    # DO NOT TOUCH, there are issues with passing nde, hpd around parallel

    # init the black box
    GLOBAL_NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_BODY_MODULES)


def init_individual(individual, id, ControllerClass, genome = None):
    # THIS SHOULD NEVER BE CALLED FROM PROCESS OTHER THAN MAIN
    # Neither should any genome be updated

    global GLOBAL_NDE, GLOBAL_HPD
    individual.id = id

    from individual import create_genome, create_body_graph,  get_body_composition

    if genome:
        individual.genome = genome
    else:
        individual.genome = create_genome()

    hpd = HighProbabilityDecoder(num_modules=NUM_BODY_MODULES)

    individual.body_graph = create_body_graph(GLOBAL_NDE, hpd, individual.genome)

    from ariel.body_phenotypes.robogen_lite.constructor import (
        construct_mjspec_from_graph,
    )
    from ariel.simulation.environments import OlympicArena

    _core = construct_mjspec_from_graph(individual.body_graph)
    _world = OlympicArena()
    _world.spawn(_core.spec, spawn_position=[1., 1., 1.])
    _model = _world.spec.compile()
    _data = mj.MjData(_model)
    individual.n_inputs = len(_data.qpos) + len(_data.qvel)
    individual.n_outputs = _model.nu

    #individual.n_inputs = len(individual.body_graph) + 3

    individual.n_cores, individual.n_joints, individual.n_bricks, individual.n_rots = get_body_composition(individual.body_graph)
    print(f'n_cores: {individual.n_cores}, n_bricks: {individual.n_bricks}, n_joints: {individual.n_joints}')


    individual.controller = ControllerClass(individual.n_inputs, individual.n_outputs)


def init_population(
        #nde: NeuralDevelopmentalEncoding,
        #hpd: HighProbabilityDecoder,
        population_size: int,
        ) -> Population:

    # return [Individual(id, controllers.RandomController) for id in range(population_size)]

    population = []
    for id in range(population_size):
        ind = Individual()
        init_individual(ind, id, controllers.NNController)
        population.append(ind)
    return population

def load_population(status: Status) -> Population | None:
    """
    load population from a location in status object or none if not available
    """

    print("loading population")
    checkpoint_dir = Path(status.checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    if not checkpoint_dir.is_dir():
        return None

    population: Population = []
    for file_path in Path(checkpoint_dir).iterdir():
        if file_path.is_file():
            population.append(load_individual(file_path))

    return population or None


def store_population(status: Status, population: Population) -> None:
    """
    store population on disk at location
    """
    checkpoint_dir = Path(status.checkpoint_dir)

    # create if doesn't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for individual in population:
        store_individual(dir_path=checkpoint_dir, individual=individual)


def tournament_selection(
        parent_population: Population,
        nr_children: int,
        arena_size: int
        ) -> Population:
    children_population = []

    for _ in range(nr_children):
        # take random individuals
        arena = random.sample(parent_population, arena_size)

        # sort by fitness (fittest first)
        arena.sort(key=lambda ind: ind.fitness, reverse=True)

        # Add best to children
        children_population.append(arena[0])

    return children_population

def mutate_population(population: Population) -> Population:
    '''
    TODO
    '''

    # TODO remember to correctly reinitialize body graphs, see init_individual

    return population
    #pass

def crossover_population(population: Population) -> Population:
    '''
    TODO
    '''

    # TODO remember to correctly reinitialize body graphs, see init_individual
    return population

def store_individual_body_graph(individual: Individual) -> None:
    '''
    thie stores the body graph as JSON, needed for hand-in
    '''
    return save_graph_as_json(
        individual.body_graph,
        "THE_BADDEST_BITCH.json"
    )

def train_individual_wrapper(individual: Individual) -> Individual:
    # EvoTorch inits the Ray clusters

    print('train individual wrapper')

    train_individual(individual, BRAIN_POPULATION_SIZE, NUM_BRAIN_ACTORS)

    return individual


def train_population(population: Population, max_workers: int) -> Population:

    print('train population')

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(train_individual_wrapper, population))

    return results

def main() -> None:
    CHECKPOINT_LOCATION = "checkpoints/"
    STATUS_LOCATION = "training_status.txt"

    init_nde_hpd()

    # load or init training status
    status: Status = None#load_training_status(STATUS_LOCATION)
    if not status:
        status = Status(
            desired_body_iterations=DEFAULT_BODY_ITERATIONS,
            current_body_iteration=0,
            checkpoint_dir=Path(CHECKPOINT_LOCATION)
        )
        store_training_status(status, STATUS_LOCATION)

    # Load or initialize Population
    population = None#load_population(status)
    if not population:
        population = init_population(BODY_POPULATION_SIZE)
        store_population(status, population)
    
    for _ in range(status.desired_body_iterations - status.current_body_iteration):
        print('main: train_population')
        population = train_population(population, max_workers=NUM_BODY_ACTORS)

        # select children, do we replace the ones we kill with new ones?
        # NOTE: when loading population from file, black box will be re-initialized
        children: Population = tournament_selection(population, BODY_POPULATION_SIZE - 1)
        children.append(Individual(id=BODY_POPULATION_SIZE))

        mutate_population(population)
        crossover_population(population)
        store_population(population)

        # update status
        display_training_status(status)
        status.current_body_iteration += 1
        store_training_status(status)

    fittest: Individual = population.sort(key=lambda ind: ind.fitness, reverse=True)[0]

    # save body as JSON
    store_individual_body_graph(fittest)

    # open an ariel window displaying fittest
    show_individual_in_window(fittest)

    # some more visualisation?
    # show_xpos_history(tracker.history["xpos"][0])


if __name__ == "__main__":
    main()
