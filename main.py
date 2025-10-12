## Standard library
from pathlib import Path
import pickle

from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding

import controllers
## Local libraries

from individual import (
    Individual,
    store_individual,
    init_individual
)

from population import (
    Population,
    evolve_population,
    init_population,
    load_population,
    store_generation,
    train_population
)

from settings import BODY_POPULATION_SIZE, NUM_BODY_ACTORS, NUM_BODY_MODULES, DEFAULT_BODY_ITERATIONS
from simulation import show_individual_in_window

from status import (
    Status,
    display_training_status,
    load_training_status,
    store_training_status
)


def store_nde(location: str, nde: NeuralDevelopmentalEncoding) -> None:
    with open(f"{location}.pkl", "wb") as f:
        pickle.dump(nde, f)

def load_nde(location: str) -> NeuralDevelopmentalEncoding | None:
    path = Path(location)

    if path.exists():
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

    ###### Main training loop ####################################
    for i in range(status.desired_body_iterations - status.current_body_iteration):
        print(f"starting generation {status.current_body_iteration}")
        population: Population = train_population(population, max_workers=NUM_BODY_ACTORS)
        print(f'length of population: {len(population)}')

        display_training_status(status)


        if status.current_body_iteration == status.desired_body_iterations - 1:
            break
        population = evolve_population(population, _init_individual)

        status.current_body_iteration += 1
        store_training_status(status, STATUS_LOCATION)

    print(f"length after training: {len(population)}")
    assert all(ind.fitness != None for ind in population)

    fittest: Individual = sorted(population, key=lambda ind: ind.fitness, reverse=True)[0]

    # pickle fittest individual
    store_individual("baddest_bitch", fittest)

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
