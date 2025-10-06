from typing import TYPE_CHECKING, Any, Literal
from ariel.simulation.environments import OlympicArena
from ariel.simulation.controllers.controller import Controller
from evotorch import Problem
import mujoco as mj
from ariel.utils.runners import simple_runner

from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

from ariel.utils.tracker import Tracker
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)

# --- DATA SETUP ---
from pathlib import Path

from individual import Fitness, Individual

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

def evaluate_individual(individual: Individual) -> Fitness:
    '''
    this would probably entail submitting the individual to
    a simulation runtime-thingy

    TODO:
        - spawn in 3 different locations
        - do abortion check
        - run simulation
        - evaluate fitness (distance)
    '''
    core: CoreModule = construct_mjspec_from_graph(individual.body_graph)
    
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"

    # not sure what this does
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )
    
    ctrl = Controller(
        controller_callback_function=individual.controller_callback,
        tracker=tracker,
    )

    run_simulation(individual, ctrl, core)

    return 0.0

def train_individual(individual: Individual) -> None:
    '''
    this is where we use evotorch Problem and CMA-ES
    assuming here that the brain is a Tensor
    '''
    problem = Problem(
        "max",

    )

    return None

def run_simulation(
        individual: Individual,
        controller: Controller,
        core: CoreModule,
        duration: int = 15
) -> None:
    '''
    to run a simulation, we only need a body and a controller.
    where we get them from though, that's the question...

    it would be nice if this 
    '''
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(core.spec, spawn_position=[0, 0, 0.1])

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
