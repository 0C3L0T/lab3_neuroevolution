## Standard library
import gc
from typing import Any

## Third party libraries
import torch
from evotorch import Problem
from optimizer_hooker import HookedSNES
from evotorch.logging import StdOutLogger
from mujoco import viewer
import mujoco as mj
import numpy as np
import copy

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

BEGIN_SPAWN_POS = [-0.8, 0, 0.1]
MIDDLE_SPAWN_POS = [1, 0, 0.1]
END_SPAWN_POS = [3, 0, 0.15]
TARGET_POS = [5, 0, 0.5]

def fitness_function(history, start_pos, target_pos) -> Fitness:
    # robot moves Flat (-1.5->-0.5) -> Rugged (~0.5->2.5) -> Inclined (~3.5 -> 4.5)
    xt, yt, zt = target_pos
    xs, ys, zs = start_pos
    xc, yc, zc = history["xpos"][0][-1]


    cartesian_distance = np.sqrt(
         (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return xc - xs

def minimal_fitness(history, start_pos) -> Fitness:
    xs, ys, zs = start_pos
    xc, yc, zc = history["xpos"][0][-1]

    return xc - xs


def run_simulation(
        controller: Controller,
        core: CoreModule,
        spawn_position: list[float],
        duration: int = 15
) -> None:
    # Initialise controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(core.spec, position=spawn_position)

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

    # This disables visualisation (fastest option)
    simple_runner(
        model,
        data,
        duration=duration,
    )

    mj.set_mjcb_control(None)

    del data
    del model
    del world
    del core

    import gc
    gc.collect()

def evaluate_individual(brain_weights: torch.Tensor, individual: Individual) -> Fitness:
    '''
    evaluate the combination of brain weights and individual in simulation
    '''
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"

    # not sure what this does
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    local_controller = copy.deepcopy(individual.controller)
    local_controller.update_weights(brain_weights)

    # TODO note that time_steps_per_save is insanely high. Make sure this does not impact fitness calculation
    ctrl = Controller(
        controller_callback_function=local_controller.callback,
        tracker=tracker,
    )

    total_fitness = 0
    for spawn_pos in [BEGIN_SPAWN_POS, MIDDLE_SPAWN_POS, END_SPAWN_POS]:
        # core needs to be recomplied every time, as attaching it to a MuJoCo works makes it dirty.
        core: CoreModule = construct_mjspec_from_graph(individual.body_graph)
        run_simulation(ctrl, core, spawn_pos)
        history = tracker.history
        total_fitness += minimal_fitness(history, spawn_pos)
        tracker.history.clear()

    average_fitness = total_fitness / 3
    return -average_fitness

def train_individual(
        individual: Individual,
        population_size: int,
        num_actors: int
        ) -> Fitness:
    '''
    train the individual CGP, return the max fitness
    '''

    xavier_bound = np.sqrt(6/(individual.n_inputs + individual.n_outputs))
    stdev_init = xavier_bound

    total_params = sum(p.numel() for p in individual.controller.parameters())

    problem = Problem(
        objective_sense="max",
        objective_func=lambda brain_weights: evaluate_individual(brain_weights, individual),
        solution_length=total_params,
        initial_bounds=(-xavier_bound, xavier_bound),
        dtype=torch.float32,
        num_actors=num_actors
    )

    aborted = False

    def stopper(i, _searcher):
        global aborted
        if i > 10 and _searcher.status['best_eval']  < 0.3:
            aborted = True
            return True
        return False

    searcher = HookedSNES(
        problem=problem,
        popsize=population_size,
        stdev_init=stdev_init,
        stopper=stopper,
    )

    # _logger = StdOutLogger(searcher)

    # TODO refactor magic number
    n_iterations = 20
    searcher.run(n_iterations)

    best_fitness = searcher.status["best_eval"]

    del searcher
    del problem
    gc.collect()

    return best_fitness


def show_individual_in_window(individual: Individual) -> None:
    '''
    This opens a live viewer of the simulation
    '''
    core: CoreModule = construct_mjspec_from_graph(individual.body_graph)
    
    # Initialise controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(core.spec, position=BEGIN_SPAWN_POS)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)
    
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"

    # not sure what this does
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    ctrl = Controller(
        controller_callback_function=individual.controller.callback,
        tracker=tracker,
    )
    
    # Pass the model and data to the tracker
    if ctrl.tracker is not None:
        ctrl.tracker.setup(world.spec, data)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)
    
    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: ctrl.set_control(m, d, *args, **kwargs),
    )

    viewer.launch(model=model, data=data)