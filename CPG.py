import torch
import torch.nn as nn

class CPG(nn.Module):
    def __init__(self, n_hinges: int, alpha=10.0, mu=1.0, omega=2*torch.pi, coupling=0.1):
        super().__init__()
        self.n = n_hinges
        self.alpha = alpha
        self.mu = mu
        self.omega = omega
        self.coupling = coupling

        # Phase offsets between oscillators (n x n matrix)
        self.phase_offsets = nn.Parameter(torch.zeros(n_hinges, n_hinges))

    def forward(self, x: torch.Tensor, y: torch.Tensor, dt=0.01):
        """
        x, y: tensors of shape (n_hinges,)
        returns next x, y
        """
        r2 = x**2 + y**2

        # Intrinsic Hopf dynamics
        dx = self.alpha * (self.mu - r2) * x - self.omega * y
        dy = self.alpha * (self.mu - r2) * y + self.omega * x

        # Coupling term (simple diff coupling)
        x_diff = x.unsqueeze(1) - x.unsqueeze(0)
        y_diff = y.unsqueeze(1) - y.unsqueeze(0)

        coupling_x = self.coupling * (x_diff * torch.cos(self.phase_offsets) - y_diff * torch.sin(self.phase_offsets)).sum(dim=1)
        coupling_y = self.coupling * (y_diff * torch.cos(self.phase_offsets) + x_diff * torch.sin(self.phase_offsets)).sum(dim=1)

        x_next = x + (dx + coupling_x) * dt
        y_next = y + (dy + coupling_y) * dt
        return x_next, y_next

def main():
    pass

if __name__ == "__main__":
    main()