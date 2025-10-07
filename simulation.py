## Standard library
from typing import Any, Callable

## Third party libraries
import torch
from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.logging import StdOutLogger
from mujoco import viewer
import mujoco as mj
import numpy as np

## Local libraries
from ariel.simulation.environments import OlympicArena
from ariel.simulation.controllers.controller import Controller
from ariel.utils.runners import simple_runner
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.utils.tracker import Tracker
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)

from individual import Fitness, Individual

# TODO figure out the start positions of each terrain
SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]

def show_individual_in_window(individual: Individual) -> None:
    '''
    This opens a live viewer of the simulation
    '''
    core: CoreModule = construct_mjspec_from_graph(individual.body_graph)
    
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(core.spec, spawn_position=SPAWN_POS)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    viewer.launch(model=model, data=data)

def fitness_function(history: list[float]) -> Fitness:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance

def evaluate_individual(individual: Individual) -> Fitness:
    '''
    this would probably entail submitting the individual to
    a simulation runtime-thingy

    TODO abortion logic
    '''
    core: CoreModule = construct_mjspec_from_graph(individual.body_graph)
    
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"

    # not sure what this does
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )
    
    # TODO construct callback function from individual NN + weights
    cpg_controller: Callable = None

    ctrl = Controller(
        controller_callback_function=cpg_controller,
        tracker=tracker,
    )

    # TODO run for every different start position
    run_simulation(individual, ctrl, core, SPAWN_POS)

    # TODO figure out how to get history
    history = None

    fitness = fitness_function(history)
    individual.fitness = fitness

    return fitness

def train_individual(
        individual: Individual,
        population_size: int,
        num_actors: int
        ) -> None:
    '''
    train the individual CGP
    '''

    # TODO do we still need this
    xavier_bound = 0

    stdev_init = xavier_bound
    
    # TODO all neurons in network (in+out+hidden)
    total_params = 24

    # what is v here?
    problem = Problem(
        objective_sense="max",
        objective_func=lambda v: evaluate_individual(v, individual),
        solution_length=total_params,
        initial_bounds=(-xavier_bound, xavier_bound),
        dtype=torch.float32,
        num_actors=num_actors
    )

    searcher = CMAES(
        problem=problem,
        popize=population_size,
        stdev_init=stdev_init
    )

    _logger = StdOutLogger(searcher)

    # TODO termination condition
    n_iterations = 100
    searcher.run(n_iterations)

    # not sure if this works
    best = searcher.status["best"]
    print(f"best candidate: {best}")
    

def run_simulation(
        individual: Individual,
        controller: Controller,
        core: CoreModule,
        spawn_position: list[float],
        duration: int = 15
) -> None:
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(core.spec, spawn_position=spawn_position)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # TODO: run the simulation for a bit and see if we can abort
    # This disables visualisation (fastest option)
    simple_runner(
        model,
        data,
        duration=duration,
    )
