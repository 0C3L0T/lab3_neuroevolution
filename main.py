## Standard library
import random
from typing import List

## Third party libraries
import numpy as np

## Local libraries
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder, save_graph_as_json

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

# Magic Numbers
NUM_BODY_MODULES = 20

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
from pathlib import Path
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# --- CUSTOM TYPES --- #
type Population = List[Individual]

def init_population(
        nde: NeuralDevelopmentalEncoding,
        hpd: HighProbabilityDecoder,
        population_size: int,
        ) -> Population:
    return [Individual(nde, hpd, id) for id in range(population_size)]

def load_population(status: Status) -> Population | None:
    """
    load population from a location in status object or none if not available
    """
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
    pass

def crossover_population(population: Population) -> Population:
    '''
    TODO
    '''
    pass

BODY_GRAPH_DIR = DATA / "bodies"
def store_individual_body_graph(individual: Individual) -> None:
    '''
    thie stores the body graph as JSON, needed for handin
    '''
    save_graph_as_json(
        individual.body_graph,
        BODY_GRAPH_DIR + str(individual.id)
    )

def main() -> None:
    POPULATION_SIZE = 8
    CHECKPOINT_LOCATION = "checkpoints/"
    STATUS_LOCATION = "training_status.txt"
    
    # init the black box
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_BODY_MODULES)
    hpd = HighProbabilityDecoder(num_modules=NUM_BODY_MODULES)

    # INIT TRAINING SPECIFICATIONS
    status: Status = load_training_status(STATUS_LOCATION)
    if not status:
        status = Status(
            desired_body_iterations=100,
            current_body_iteration=0,
            checkpoint_dir=Path(CHECKPOINT_LOCATION)
        )
        store_training_status(status, STATUS_LOCATION)

    # Load or initialize Population
    population = load_population(status)
    if not population:
        population = init_population(nde, hpd, POPULATION_SIZE)
        store_population(status, population)
    
    # Load or initialize population brains?
        # The brains need to be stored together with the bodies that they belong to
        # maybe even as we init the bodies
        # currently the bodies are already stored with brains

    for _ in range(status.desired_body_iterations - status.current_body_iteration):

        # train the brains
        for individual in population:
            # initialize brain (controller specific)
            # in the NN case, every body gets a custom controller
            # here i'm assuming the brain is a Tensor
            # TODO

            # train brain (controller specific)
                # every individual NN controller gets trained
            train_individual(individual)

        # select children, do we replace the ones we kill with new ones?
        children = tournament_selection(population, POPULATION_SIZE - 1)
        children.append(Individual(id=POPULATION_SIZE, nde=nde, hpd=hpd))

        # mutate/crossover bodies
            # this is why we need to store the Genome in the Individual
            # also don't know if we want to do this manually or not
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
