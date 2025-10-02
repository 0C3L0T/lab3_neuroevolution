
## Standard library
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

## Third party libraries
import mujoco as mj
import numpy as np
import torch

## Local libraries
from ariel.simulation.controllers.controller import Controller
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)

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
@dataclass
class Individual:
    id: int
    genome: Genome
    body_graph: DiGraph[Any]
    brain: torch.Tensor
    fitness: Fitness


def initialize_individual_brain(individual: Individual) -> None:
    return None

def create_individual_body(
        nde: NeuralDevelopmentalEncoding,
        hpd: HighProbabilityDecoder
        ) -> DiGraph[Any]:
    """
    Create a random robot graph using the NDE.
    NOTE: A graph is not yet a body

    TODO
    - store the layout of the body somewhere so we can init the brains later
    """
    type_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
    conn_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
    rot_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)

    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]

    p_matrices = nde.forward(genotype)

    # Decode the high-probability graph
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    return robot_graph

def create_individual(nde: NeuralDevelopmentalEncoding, hpd: HighProbabilityDecoder) -> Individual:
    pass

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

    return 0.0

def train_individual(individual: Individual) -> None:
    '''
    this is where we use evotorch Problem and CMA-ES
    assuming here that the brain is a Tensor
    '''

    return None


BODY_GRAPH_DIR = DATA / "bodies"
def store_individual_body_graph(individual: Individual) -> None:
    save_graph_as_json(
        individual.body_graph,
        BODY_GRAPH_DIR + str(individual.id)
    )