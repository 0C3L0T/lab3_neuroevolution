import torch
from evotorch import Problem

def pairwise_distances(positions: torch.Tensor) -> torch.Tensor:
    positions = positions.view(positions.shape[0], -1, 3)
    deltas = positions.unsqueeze(2) - positions.unsqueeze(1)
    distances = torch.norm(deltas, dim=-1)
    return distances

def cluster_potential(positions: torch.Tensor) -> torch.Tensor:
    distances = pairwise_distances(positions)
    pairwise_cost = (1 / distances).pow(12) - (1 / distances).pow(6.0)
    ut_pairwise_cost = torch.triu(pairwise_cost, diagonal=1)
    potential = 4 * ut_pairwise_cost.sum(dim=(1, 2))
    return potential

def get_sample_problem(num_actors=1):
    # Minimize the Lennard-Jones atom cluster potential
    problem = Problem(
        "min",
        cluster_potential,
        initial_bounds=(-1e-12, 1e-12),
        #device="cuda:0" if torch.cuda.is_available() else "cpu",
        device="cpu",
        solution_length=150,
        # Evaluation is vectorized
        vectorized=True,
        # Higher-than-default precision
        dtype=torch.float64,
        num_actors=num_actors,

    )

    return problem