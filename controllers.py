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
        self.net = nn.Linear(n_inputs, n_outputs)

        self._initialize_weights()


    def _initialize_weights(self):
        nn.init.xavier_uniform(self.net.weight)
        nn.init.zeros_(self.net.bias)

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

