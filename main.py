import os
from pathlib import Path
import pickle
from typing import Callable, List
from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding
from evotorch import Problem, Solution

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
    DEFAULT_BODY_ITERATIONS,
    GENOTYPE_SIZE,
    NDE_LOCATION,
    NUM_BODY_ACTORS,
    NUM_BRAIN_ACTORS,
)

from simulation import show_individual_in_window, train_individual

def store_generation(generation: int, solver: SNES, init_individual_function: Callable[[], Individual]) -> None:
    pop = solver.population

    checkpoint_dir = Path(f"{CHECKPOINT_LOCATION}/generation_{generation}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for sol in pop:
        # create Individual from genome using NDE?
        genome = sol.values

        individual: Individual = init_individual_function(genome)
        store_individual(checkpoint_dir, individual)


def evaluate_genome(body_genome: Genome, init_individual_function: Callable[[], Individual]) -> Fitness:
    individual: Individual = init_individual_function(body_genome)
    genome_fitness: Fitness = train_individual(individual, BRAIN_POPULATION_SIZE, NUM_BRAIN_ACTORS)
    return genome_fitness

def main() -> None:
    nde: NeuralDevelopmentalEncoding = load_or_init_nde(NDE_LOCATION)

    # centralized definition. we use one NDE and one controller type
    _init_individual = lambda genome: init_individual(nde, controllers.lobotomizedCPG, genome)

    CHECKPOINT_PATH = "save_state.pt"
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming solver from {CHECKPOINT_PATH}")
        body_solver = SNES.load(CHECKPOINT_PATH)
    else:
        print("Starting new SNES run")
        BODY_DIMENSION = 3 * GENOTYPE_SIZE
        body_problem: Problem = Problem(
            objective_sense="max",
            objective_func=lambda genome: evaluate_genome(genome, _init_individual),
            solution_length=BODY_DIMENSION,
            initial_bounds=(0, 1),
            dtype=torch.float32,
            num_actors=NUM_BODY_ACTORS
        )

        body_solver = SNES(
            problem=body_problem,
            popsize=BODY_POPULATION_SIZE,
            stdev_init=0.2
        )
    

    StdOutLogger(body_solver)

    ## MAIN TRAINING LOOP
    for generation in range(DEFAULT_BODY_ITERATIONS):
        print(f"\n==== Generation {generation} ====")
        body_solver.run(1)
        body_solver.save(CHECKPOINT_PATH)
        store_generation(generation, body_solver, _init_individual)
        
    print("END OF TRAINING")

    fittest_genome: Solution = body_solver.status["best"]
    fittest_individual: Individual = _init_individual(fittest_genome.values)

    # open an ariel window displaying fittest
    try:
        show_individual_in_window(fittest_individual)
    except Exception as e:
        print(f"[WARN] Could not open Ariel window for fittest individual: {e}")

    # show_xpos_history(tracker.history["xpos"][0])


if __name__ == "__main__":
    main()
