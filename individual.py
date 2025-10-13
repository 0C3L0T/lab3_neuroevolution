
## Standard library
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
import pickle
from typing import List 

## Third party libraries
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.environments.olympic_arena import OlympicArena
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import mujoco as mj

## Local libraries
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder, save_graph_as_json
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding

import torch
import torch.nn as nn
import numpy as np

import controllers
from settings import GENOTYPE_SIZE, NUM_BODY_MODULES

# list of 3 alleles
type Genome = List[np.float32]

type Fitness = float

@dataclass
class Individual:
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
    fitness: float = 0

def init_individual(
    nde: NeuralDevelopmentalEncoding,
    ControllerClass: nn.Module,
    unsplit_genome: torch.Tensor,
) -> Individual:

    # split genome in three arrays
    unsplit_np =unsplit_genome.detach().cpu().tolist()
    genome: np.ndarray = [unsplit_np[i*GENOTYPE_SIZE:(i+1)*GENOTYPE_SIZE] for i in range(3)]
    genome = np.array(genome)

    # --- Body generation ---
    hpd = HighProbabilityDecoder(num_modules=NUM_BODY_MODULES)
    body_graph = create_body_graph(nde, hpd, genome)

    # --- MuJoCo model construction ---
    core_spec = construct_mjspec_from_graph(body_graph)
    world = OlympicArena()
    world.spawn(core_spec.spec, position=[1.0, 1.0, 1.0])
    model = world.spec.compile()
    data = mj.MjData(model)

    # --- IO and structure composition ---
    n_inputs = ControllerClass.num_inputs(len(data.qpos), len(data.qvel), 1)
    n_outputs = model.nu
    n_cores, n_joints, n_bricks, n_rots = get_body_composition(body_graph)

    # --- Controller ---
    controller = ControllerClass(n_inputs, n_outputs)

    id = np.random.randint(10000000)

    del data
    del model
    del world
    import gc
    gc.collect()
    mj.set_mjcb_control(None)

    # --- Final assembly of Individual dataclass ---
    return Individual(
        id=id,
        genome=genome,
        body_graph=body_graph,
        controller=controller,
        n_cores=n_cores,
        n_bricks=n_bricks,
        n_joints=n_joints,
        n_rots=n_rots,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
    )

def display_individual(individual):
    """Pretty-print all fields of an Individual dataclass."""
    assert isinstance(individual, Individual), "Expected an Individual instance"
    print("── Individual ──")
    for f in fields(individual):
        name = f.name
        value = getattr(individual, name)
        print(f"{name:>12}: {value}")


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

def store_individual(dir_path: str, individual: Individual) -> None:
    '''
    pickle an Individual, assume the dir path exists
    '''
    assert is_dataclass(individual) and isinstance(individual, Individual), \
    "Expected an instance of Individual dataclass"

    # create dir path if not exist
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    with open(f"{dir_path}/{individual.id}.pkl", "wb") as f:
        pickle.dump(individual, f)
        

def store_individual_body_graph(individual: Individual) -> None:
    '''
    thie stores the body graph as JSON, needed for hand-in
    '''
    return save_graph_as_json(
        individual.body_graph,
        "THE_BADDEST_BITCH.json"
    )
    
def main():
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_BODY_MODULES)
    _init_individual = lambda **kwargs: init_individual(nde, controllers.lobotomizedCPG, **kwargs)
    i = _init_individual()

    display_individual(i)

if __name__ == "__main__":
    main()