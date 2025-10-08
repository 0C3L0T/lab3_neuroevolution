
## Standard library
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
import pickle
from typing import List 

## Third party libraries
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph

## Local libraries
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
import torch

import torch.nn as nn
from CPG import CPG

import controllers

# MAGIC NUMBERS
GENOTYPE_SIZE = 64


# --- RANDOM GENERATOR SETUP --- #
#SEED = 42
RNG = np.random.default_rng()#SEED)

# list of 3 alleles
type Genome = List[np.float32]

type Fitness = float

@dataclass
class Individual:
    '''
    TODO have a fitness property
    
    '''
    id: int | None = None
    genome: Genome | None = None
    body_graph: nx.DiGraph | None = None
    controller: nn.Module | None = None
    n_cores: int | None = None
    n_bricks: int | None = None
    n_joints: int | None = None
    n_rots: int | None = None
    n_inputs: int | None = None
    n_outputs: int | None = None

def display_individual(individual):
    """Pretty-print all fields of an Individual dataclass."""
    assert isinstance(individual, Individual), "Expected an Individual instance"
    print("── Individual ──")
    for f in fields(individual):
        name = f.name
        value = getattr(individual, name)
        print(f"{name:>12}: {value}")

def create_genome():
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
        genome
        ) -> nx.DiGraph:
    """
    NOTE: A graph is not yet a body
    """
    p_matrices = nde.forward(genome)

    # Decode the high-probability graph
    robot_graph: nx.DiGraph = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    return robot_graph

def get_body_composition(body_graph: nx.DiGraph):
    data = json_graph.node_link_data(body_graph, edges="edges")

    nodes = data.get("nodes", [])

    return (sum(1 for node in nodes if node.get("type") == node_type)
            for node_type in ['CORE', 'HINGE', 'BRICK', 'NONE'])

def load_individual(path: Path) -> Individual:
    '''
    unpickle an Individual, assume path exists
    '''
    with open(path, "rb") as f:
        return pickle.load(f)

def store_individual(dir_path: Path, individual: Individual) -> None:
    '''
    pickle an Individual, assume the dir path exists
    '''
    assert is_dataclass(individual) and isinstance(individual, Individual), \
    "Expected an instance of Individual dataclass"

    with open(f"{dir_path}/{individual.id}.pkl", "wb") as f:
        pickle.dump(individual, f)
        
    
def main():
    i: Individual = Individual()
    i.id = 5

    display_individual(i)

if __name__ == "__main__":
    main()