
## Standard library
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List

## Third party libraries
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.utils.tracker import Tracker
import mujoco as mj
import numpy as np
import torch
from evotorch import Problem

## Local libraries
from ariel.simulation.controllers.controller import Controller
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)

from simulation import run_simulation

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# MAGIC NUMBERS
GENOTYPE_SIZE = 64

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

type Genome = List[np.float32]
type Fitness = float

'''
TODO
- can we use the built-in Individual class?
- write constructor, or keep imperative? 
'''
class Individual:
    id: int
    genome: Genome
    body_graph: DiGraph[Any]
    network: torch.nn
    weights: torch.Tensor
    fitness: Fitness
    controller_callback: Callable

    def __init__(
            self,
            nde: NeuralDevelopmentalEncoding,
            hpd: HighProbabilityDecoder,
            id: int,
            ):
        self.id = id
        self.genome = create_genome()
        self.body_graph = create_body_graph(nde, hpd, self.genome)

        # figure out model input/ouput
            # is there a way to get these without initialising mujoco?
        
        # init NN

        # init weights




def create_genome(void) -> Genome:
    type_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
    conn_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
    rot_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)

    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]

    return genotype


def create_body_graph(
        nde: NeuralDevelopmentalEncoding,
        hpd: HighProbabilityDecoder,
        genome: Genome
        ) -> DiGraph[Any]:
    """
    NOTE: A graph is not yet a body
    """
    p_matrices = nde.forward(genome)

    # Decode the high-probability graph
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    return robot_graph


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


BODY_GRAPH_DIR = DATA / "bodies"
def store_individual_body_graph(individual: Individual) -> None:
    save_graph_as_json(
        individual.body_graph,
        BODY_GRAPH_DIR + str(individual.id)
    )