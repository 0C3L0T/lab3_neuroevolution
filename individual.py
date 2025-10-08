
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

import controllers

# MAGIC NUMBERS
GENOTYPE_SIZE = 64


# --- RANDOM GENERATOR SETUP --- #
#SEED = 42
RNG = np.random.default_rng()#SEED)

type Genome = List[np.float32]
type Fitness = float

class Individual:
    # Individual has a set body, but a variable brain
    # id: int
    # # genome = None: Genome
    # body_graph: nx.DiGraph
    # controller = None
    #fitness: Fitness = 0
    # cpg: CPG
    id = None
    genome = None
    body_graph = None
    controller = None

    # def __init__(
    #         self,
    #         id: int,
    #         #nde: NeuralDevelopmentalEncoding,
    #         #hpd: HighProbabilityDecoder,
    #         ControllerClass,
    #         ):
        # self.id = id
        #
        # NUM_OF_MODULES = 30
        #
        # from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
        # from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder, \
        #     save_graph_as_json
        # genotype_size = 64
        # type_p_genes = RNG.random(genotype_size).astype(np.float32)
        # conn_p_genes = RNG.random(genotype_size).astype(np.float32)
        # rot_p_genes = RNG.random(genotype_size).astype(np.float32)
        #
        # self.genotype = [
        #     type_p_genes,
        #     conn_p_genes,
        #     rot_p_genes,
        # ]
        #
        # # nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
        # p_matrices = nde.forward(self.genotype)
        #
        # # Decode the high-probability graph
        # # hpd = HighProbabilityDecoder(NUM_OF_MODULES)
        # self.body_graph = hpd.probability_matrices_to_graph(
        #     p_matrices[0],
        #     p_matrices[1],
        #     p_matrices[2],
        # )

        #self.body_graph = create_body_graph(nde, hpd, self.genome)

        #n_cores =
        #n_joints = count_joints_in_body(self.body_graph)
        # self.n_cores, self.n_joints, self.n_bricks, self.n_rots = get_body_composition(self.body_graph)
        #
        # # TODO move *6 in ControllerClass
        # self.controller = ControllerClass(self.n_cores * 6, self.n_joints)

        # TODO find sensible defaults, do we store these parameters in Individual?
        # self.cpg = CPG(
        #     n_hinges=n_joints,
        #     alpha=10.0,
        #     mu=1.0,
        #     omega=2*np.pi,
        #     coupling=0.1
        # )



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
    print(data)

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