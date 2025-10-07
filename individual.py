
## Standard library
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

from CPG import CPG

# MAGIC NUMBERS
GENOTYPE_SIZE = 64

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

type Genome = List[np.float32]
type Fitness = float

class Individual:
    id: int
    genome: Genome
    body_graph: nx.DiGraph
    fitness: Fitness = 0
    cpg: CPG

    def __init__(
            self,
            nde: NeuralDevelopmentalEncoding,
            hpd: HighProbabilityDecoder,
            id: int,
            ):
        self.id = id
        self.genome = create_genome()
        self.body_graph = create_body_graph(nde, hpd, self.genome)

        n_joints = count_joints_in_body(self.body_graph)

        # TODO find sensible defaults, do we store these parameters in Individual?
        self.cpg = CPG(
            n_hinges=n_joints,
            alpha=10.0,
            mu=1.0,
            omega=2*np.pi,
            coupling=0.1
        )

    def __str__(self) -> str:
        return (
            f"Individual(id={self.id}, "
            f"genome={type(self.genome).__name__}, "
            f"fitness={self.fitness}, "
            f"nodes={len(self.body_graph.nodes)}, "
            f"edges={len(self.body_graph.edges)}, "
            # f"network={self.network.__class__.__name__}, "
            # f"weights_shape={tuple(self.weights.shape)}, "
            # f"controller_callback={self.controller_callback.__name__ if hasattr(self.controller_callback, '__name__') else type(self.controller_callback).__name__})"
        )


def create_genome() -> Genome:
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

def count_joints_in_body(body_graph: nx.DiGraph) -> int:
    data = json_graph.node_link_data(body_graph, edges="edges")

    nodes = data.get("nodes", [])
    return sum(1 for node in nodes if node.get("type") == "HINGE")

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
    with open(f"{dir_path}/{individual.id}.pkl", "wb") as f:
        pickle.dump(individual, f)
        
    

def main():
    hpd = HighProbabilityDecoder(20)
    nde = NeuralDevelopmentalEncoding(20)
    i: Individual = Individual(hpd=hpd, nde=nde, id=0)


    # store_individual(".", i)
    # i2 = load_individual("0.pkl")

    # Path("0.pkl").unlink()

    # print(i)
    # print(i2)

if __name__ == "__main__":
    main()