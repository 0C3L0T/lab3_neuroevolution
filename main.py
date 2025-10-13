from pathlib import Path
from typing import Callable
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import RNG
from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding
from evotorch import Problem

import torch
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger

## Local Libraries
from NDE import load_or_init_nde
import controllers
from individual import (
    Fitness,
    Genome,
    Individual,
    init_individual,
    store_individual
)

from settings import (
    BODY_POPULATION_SIZE,
    BRAIN_POPULATION_SIZE,
    CHECKPOINT_LOCATION,
    GENOTYPE_SIZE,
    NDE_LOCATION,
    NUM_BODY_ACTORS,
    NUM_BRAIN_ACTORS,
    STATUS_LOCATION
)

from simulation import show_individual_in_window, train_individual
from status import Status, load_or_init_status, store_training_status

def store_generation(generation: int, solver: SNES, init_individual_function: Callable[[], Individual]) -> None:
    pop = solver.population

    checkpoint_dir = Path(f"{CHECKPOINT_LOCATION}/generation_{generation}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for sol in pop:
        # create Individual from genome using NDE?
        genome = sol.values

        # TODO check format
        print(genome)
        individual: Individual = init_individual_function(genome)
        store_individual(checkpoint_dir, individual)


def evaluate_genome(body_genome: Genome, init_individual_function: Callable[[], Individual]) -> Fitness:
    individual: Individual = init_individual_function(body_genome)
    genome_fitness: Fitness = train_individual(individual, BRAIN_POPULATION_SIZE, NUM_BRAIN_ACTORS)
    return genome_fitness

def main() -> None:
    nde: NeuralDevelopmentalEncoding = load_or_init_nde(NDE_LOCATION)
    status: Status = load_or_init_status(STATUS_LOCATION)

    # centralized definition. we use one NDE and one controller type
    _init_individual = lambda **kwargs: init_individual(nde, controllers.lobotomizedCPG, **kwargs)

    '''
    TODO
        - check if we can load an initial population
        - set initial bounds according to normal distribution?
    '''
    BODY_DIMENSION = 3 * GENOTYPE_SIZE
    body_problem: Problem = Problem(
        objective_sense="max",
        objective_func=lambda genome: evaluate_genome(genome, _init_individual),
        solution_length=BODY_DIMENSION,
        initial_bounds=RNG.random(BODY_DIMENSION).astype(torch.float32),
        dtype=torch.float32,
        num_actors=NUM_BODY_ACTORS
    )

    body_solver = SNES(body_problem, popsize=BODY_POPULATION_SIZE, stdev_init=0.2)
    StdOutLogger(body_solver)

    ## MAIN TRAINING LOOP
    for generation in range(status.desired_body_iterations - status.current_body_iteration):
        print(f"\n==== Generation {generation} ====")

        # TODO change
        store_generation(generation, body_solver)
        body_solver.run(1)

        print(f"current best fitness: {body_solver.status["best_fitness"]}")
        
        status.current_body_iteration += 1
        store_training_status(status, STATUS_LOCATION)

    fittest_genome: Genome = body_solver.status["best"]
    
    # TODO check if this format is correct
    print(fittest_genome)

    fittest_individual: Individual = _init_individual(fittest_genome)

    # open an ariel window displaying fittest
    try:
        show_individual_in_window(fittest_individual)
    except Exception as e:
        print(f"[WARN] Could not open Ariel window for fittest individual: {e}")

    # show_xpos_history(tracker.history["xpos"][0])


if __name__ == "__main__":
    main()
