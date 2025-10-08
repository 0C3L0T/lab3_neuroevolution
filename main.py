## Standard library
from concurrent.futures import ProcessPoolExecutor
import os
import pickle
import random
from typing import List

## Third party libraries
import torch.nn as nn
import mujoco as mj

## Local libraries
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder, save_graph_as_json
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.environments.olympic_arena import OlympicArena

from individual import (
    Genome,
    Individual,
    create_body_graph,
    create_genome,
    get_body_composition,
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
NUM_BRAIN_ACTORS = os.cpu_count() // 2
NUM_BODY_ACTORS = 2
BODY_POPULATION_SIZE = 100
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

def init_individual(
    id: int,
    ControllerClass: nn.Module,
    genome: Genome | None = None
) -> Individual:
    """
    Initialize a new Individual.
    NOTE: Should only be called from the main process.
    """
    # get NDE and HPD into scope
    global GLOBAL_NDE

    # --- Genome setup ---
    genome = genome or create_genome()

    # --- Body generation ---
    hpd = HighProbabilityDecoder(num_modules=NUM_BODY_MODULES)
    body_graph = create_body_graph(GLOBAL_NDE, hpd, genome)

    # --- MuJoCo model construction ---
    core_spec = construct_mjspec_from_graph(body_graph)
    world = OlympicArena()
    world.spawn(core_spec.spec, spawn_position=[1.0, 1.0, 1.0])
    model = world.spec.compile()
    data = mj.MjData(model)

    # --- IO and structure composition ---
    n_inputs = ControllerClass.num_inputs(len(data.qpos), len(data.qvel), 1)
    n_outputs = model.nu
    n_cores, n_joints, n_bricks, n_rots = get_body_composition(body_graph)

    print(f"n_cores: {n_cores}, n_bricks: {n_bricks}, n_joints: {n_joints}")

    # --- Controller ---
    controller = ControllerClass(n_inputs, n_outputs)

    # --- Final assembly of Individual dataclass ---
    return Individual(
        id=id,
        genome=genome,
        body_graph=body_graph,
        controller=controller,
        n_cores=n_cores,
        n_bricks=n_bricks,
        n_joints=n_joints,
        n_rots=n_rots,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
    )

def init_population(population_size: int) -> Population:
    return [init_individual(id, controllers.NNController) for id in range(population_size)]

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

def load_nde(location: str) -> NeuralDevelopmentalEncoding:
    path = Path(location)

    if not path.exists():
        return None
    
    #unpickle
    with open(f"{path}.pkl", "rb") as f:
        return pickle.load(f)

def store_nde(location: str, nde: NeuralDevelopmentalEncoding) -> None:
    with open(f"{location}.pkl", "wb") as f:
        pickle.dump(nde, f)


def store_generation(status: Status, population: Population, generation: int) -> None:
    """
    store generation population on disk
    """
    checkpoint_dir = Path(f"{status.checkpoint_dir}/generation_{generation}")

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

def crossover_gene(gene_1: Genome, gene_2: Genome, mutation_rate: float) -> None:
    for allele in range(len(gene_1)):
        for (a, b) in zip(gene_1[allele], gene_2[allele]):
            if random.random < mutation_rate:
                t = a
                a = b
                b = t


def crossover_population(population: Population, mutation_rate: float) -> None:
    '''
    crossover population genomes in-place and pair wise.
    '''
    assert(len(population) % 2 == 0)

    
    
    for i in range(0, len(population), 2):
        individual_1 = population[i]
        individual_2 = population[i+1]
        crossover_gene(individual_1.genome, individual_2.genome, mutation_rate)


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
    NDE_LOCATION = "NDE"

    # load or init global NDE
    global GLOBAL_NDE
    nde = load_nde(NDE_LOCATION)
    if not nde:
        nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_BODY_MODULES)
        store_nde(NDE_LOCATION, nde)
    GLOBAL_NDE = nde


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
        store_generation(status, population, generation=0)
    
    for _ in range(status.desired_body_iterations - status.current_body_iteration):
        print('main: train_population')
        population = train_population(population, max_workers=NUM_BODY_ACTORS)

        # select children, do we replace the ones we kill with new ones?
        # NOTE: when loading population from file, black box will be re-initialized
        children: Population = tournament_selection(population, BODY_POPULATION_SIZE - 1)
        children.append(Individual(id=BODY_POPULATION_SIZE))

        mutate_population(population)
        crossover_population(population)
        store_generation(status, population, generation=status.current_body_iteration)

        # update status
        display_training_status(status)
        status.current_body_iteration += 1
        store_training_status(status)

    fittest: Individual = population.sort(key=lambda ind: ind.fitness, reverse=True)[0]

    # save body as JSON
    store_individual_body_graph(fittest)

    # open an ariel window displaying fittest
    try:
        show_individual_in_window(fittest)
    except Exception as e:
        print(f"[WARN] Could not open Ariel window for fittest individual: {e}")


    # TODO video functionality

    # some more visualisation?
    # show_xpos_history(tracker.history["xpos"][0])


if __name__ == "__main__":
    main()
