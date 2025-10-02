## Standard library
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

## Third party libraries
import numpy as np
import torch

## Local libraries
from ariel.simulation.controllers.controller import Controller
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder

from individual import (
    Individual,
    Genome,
    Fitness,
    create_individual,
    create_individual_body,
    initialize_individual_brain,
    train_individual,
)

from status import (
    Status,
    display_training_status,
    load_training_status,
    update_training_status
)

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

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

# TODO: global load/store mechanism

# --- CUSTOM TYPES --- #
type Population = List[Individual]

def init_population(
        nde: NeuralDevelopmentalEncoding,
        hpd: HighProbabilityDecoder,
        population_size: int,
        ) -> Population:

    population = []
    for _ in range(population_size):
        population.append(create_individual_body(nde, hpd))

    return population

def load_population(location: Path) -> Population | None:
    """
    return population from specified location, or None if not available
    """
    # Some try/catch logic

def store_population(location: Path, population: Population) -> None:
    """
    store population on disk at location
    """
    # some try/catch logic

def main() -> None:
    POPULATION_SIZE = 8
    
    # init the NDE and HPD
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_BODY_MODULES)
    hpd = HighProbabilityDecoder(num_modules=NUM_BODY_MODULES)

    # Load or initialize population bodies
    population = load_population()
    if not population:
        population = init_population(nde, hpd, POPULATION_SIZE)
        store_population(population)

    # Load or initialize population brains
        # The brains need to be stored together with the bodies that they belong to
        # maybe even as we init the bodies


    # TODO: Keep track of trainings status
    status: Status = load_training_status()

    for _ in range(status.desired_body_iterations - status.current_body_iteration):

        # train the brains
        for individual in population:
            # initialize brain (controller specific)
            # in the NN case, every body gets a custom controller
            # here i'm assuming the brain is a Tensor
            initialize_individual_brain(individual)

            # train brain (controller specific)
                # every individual NN controller gets trained
            train_individual(individual)

        # select winners
            # this is just the ones with highest fitness
            # do we remove some and replace with new ones?
        
        # sort population in descending order (highest first)
        population = population.sort(key=lambda ind: ind.fitness, reverse=True)

        # replace lowest individual?
        population[-1] = create_individual(nde, hpd)

        # mutate/crossover bodies
            # this is why we need to store the Genome in the Individual
            # also don't know if we want to do this manually or not

        store_population(population)

        # update status
        display_training_status(status)
        status.current_body_iteration += 1
        update_training_status(status)

    # here we can run an interactive window with the best individual


    # show_xpos_history(tracker.history["xpos"][0])


if __name__ == "__main__":
    main()
