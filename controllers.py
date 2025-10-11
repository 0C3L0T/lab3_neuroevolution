import torch
import numpy as np
import torch.nn as nn



def vector_to_params(vec, net):
    param_tensors = [p.detach().clone() for p in net.parameters()]
    shapes = [p.shape for p in param_tensors]
    sizes = [p.numel() for p in param_tensors]
    total_params = sum(sizes)

    idx = 0
    with torch.no_grad():
        for p, n in zip(net.parameters(), sizes):
            p.copy_(vec[idx:idx + n].view_as(p))
            idx += n


class NNController(nn.Module):
    
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 2 * 16),
            #nn.Tanh(),
            #nn.Linear(2 * 16, 2 * 16),
            nn.Tanh(),
            nn.Linear(2 * 16, n_outputs)
        )

    def forward(self, x):
        return torch.tanh(self.net(x)) * (np.pi / 2)

    def callback(self, m, d):
        inputs = torch.from_numpy(
            np.concatenate([
                d.qpos,
                d.qvel
            ])
        ).to(
            torch.float32
        )

        #print('d.qpos.shape, d.qvel.shape', d.qpos.shape, d.qvel.shape)

        with torch.no_grad():
            outputs = self.forward(inputs)

        return outputs

    def update_weights(self, v):
        vector_to_params(v, self)

    @staticmethod
    def num_inputs(n_qpos, n_qvel, n_time):
        return n_qpos + n_qvel


class lobotomizedCPG(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()

        self.A = nn.Parameter(torch.zeros(n_outputs))
        self.omega = nn.Parameter(torch.zeros(n_outputs))
        self.phi = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, x):
        t = x[-1]

        return self.A * torch.sin(self.omega * t + self.phi)

    def callback(self, m, d):
        inputs = torch.tensor([d.time], dtype=torch.float32)
        with torch.no_grad():
            outputs = self.forward(inputs)
        return outputs

    def update_weights(self, v):
        vector_to_params(v, self)

    @staticmethod
    def num_inputs(n_qpos, n_qvel, n_time):
        return n_time