## Standard library
from pathlib import Path
import pickle

from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding

import controllers
## Local libraries

from individual import (
    Individual,
    store_individual_body_graph,
    init_individual
)

from population import (
    Population,
    evolve_population,
    init_population,
    load_population,
    mutate_crossover_population,
    store_generation,
    tournament_selection,
    train_population
)

from simulation import show_individual_in_window

from status import (
    Status,
    display_training_status,
    load_training_status,
    store_training_status
)

# Magic Numbers
NUM_BODY_MODULES = 30  # they've changed this to 30 in the template now
# TODO: make sure to modify this before submitting to AWS
NUM_BODY_ACTORS = 2
BODY_POPULATION_SIZE = 6 # should be multiple of 6
DEFAULT_BODY_ITERATIONS = 3

# --- RANDOM GENERATOR SETUP --- #

#SEED = 42
#RNG = np.random.default_rng(SEED)

def store_nde(location: str, nde: NeuralDevelopmentalEncoding) -> None:
    with open(f"{location}.pkl", "wb") as f:
        pickle.dump(nde, f)

def load_nde(location: str) -> NeuralDevelopmentalEncoding | None:
    path = Path(location)

    if not path.exists():
        return None
    
    with open(f"{path}.pkl", "rb") as f:
        return pickle.load(f)

def main() -> None:
    CHECKPOINT_LOCATION = "checkpoints/"
    STATUS_LOCATION = "training_status.txt"
    NDE_LOCATION = "NDE"

    # Load or init NDE
    nde = load_nde(NDE_LOCATION)
    if not nde:
        print("CREATING NEW NDE")
        nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_BODY_MODULES)
        store_nde(NDE_LOCATION, nde)

    # centralized definition. we use one NDE and one controller type
    _init_individual = lambda **kwargs: init_individual(nde, controllers.lobotomizedCPG, **kwargs)

    # Load or init training status
    status: Status = load_training_status(STATUS_LOCATION)
    if not status:
        print("CREATING NEW STATUS")
        status = Status(
            desired_body_iterations=DEFAULT_BODY_ITERATIONS,
            current_body_iteration=0,
            checkpoint_dir=Path(CHECKPOINT_LOCATION)
        )
        store_training_status(status, STATUS_LOCATION)

    # Load or initialize Population
    population = load_population(status)
    if not population:
        print("CREATE NEW POPULATION")

        population = init_population(BODY_POPULATION_SIZE, _init_individual)
        store_generation(status, population, generation=0)

    print('population leng', len(population))
    assert(len(population) % 2 == 0)
    
    ###### Main training loop ####################################
    for _ in range(status.desired_body_iterations - status.current_body_iteration):
        print(f"starting generation {status.current_body_iteration}")
        population: Population = train_population(population, max_workers=NUM_BODY_ACTORS)

        population = evolve_population(population, _init_individual)

        store_generation(status, population, generation=status.current_body_iteration)
        display_training_status(status)
        status.current_body_iteration += 1
        store_training_status(status, STATUS_LOCATION)

    fittest: Individual = sorted(population, key=lambda ind: ind.fitness, reverse=True)[0]

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
