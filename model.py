import torch
from torch import nn


class Viewer(nn.Module):
    def __init__(self, P, H):
        super(Viewer, self).__init__()
        self.lstm = nn.LSTM(P + 1, H)
        self.h0 = nn.Parameter(torch.randn(1, 1, H), requires_grad=True)
        self.c0 = nn.Parameter(torch.randn(1, 1, H), requires_grad=True)

    def forward(self, view):
        """
        INPUTS
        view: (B, N, P+1)

        OUTPUTS
        tuple (outputs, ((1, B, H), (1, B, H)))
        """
        B = view.shape[0]
        view = view.transpose(0, 1)
        h0 = self.h0.repeat(1, B, 1)
        c0 = self.c0.repeat(1, B, 1)
        return self.lstm(view, (h0, c0))


class Communicator(nn.Module):
    def __init__(self, P, H, M):
        super(Communicator, self).__init__()
        self.fc1 = nn.Linear(H, M)
        self.fc2 = nn.Linear(M, M)
        self.lstm = nn.LSTM(P * M, H)

    def forward(self, messages, hc_t):
        """
        INPUTS
        messages: B x (P * M)
        hc_t: tuple (h_t, c_t)

        OUTPUTS
        tuple (message, hc_t1)
        message: (B, M) messages to other agents
        hc_t1: tuple (h_t1, c_t1)
        """
        _, hc_t1 = self.lstm(messages[None, :, :], hc_t)
        return self.get_message(hc_t1[0]), hc_t1

    def get_message(self, h):
        x = torch.relu(self.fc1(h))
        return torch.tanh(self.fc2(x))


class Voter(nn.Module):
    def __init__(self, P, H):
        super(Voter, self).__init__()
        self.fc1 = nn.Linear(2 * H, H)
        self.fc2 = nn.Linear(H, P)
        self.softmax = nn.Softmax(2)

    def forward(self, hc_n):
        x = torch.cat(hc_n, dim=2)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)[0]


class Agent:
    def __init__(self, device, P, H, M):
        self.viewer = Viewer(P, H)
        self.comm = Communicator(P, H, M)
        self.vote = Voter(P, H)

        self.viewer.to(device)
        self.comm.to(device)
        self.vote.to(device)

    def requires_grad_(self, rg):
        rg = bool(rg)
        self.viewer.requires_grad_(rg)
        self.comm.requires_grad_(rg)
        self.vote.requires_grad_(rg)

    def parameters(self):
        return [
            {"params": self.viewer.parameters()},
            {"params": self.comm.parameters()},
            {"params": self.vote.parameters()},
        ]

    def copy_state(self, other):
        self.viewer.load_state_dict(other.viewer.state_dict())
        self.comm.load_state_dict(other.comm.state_dict())
        self.vote.load_state_dict(other.vote.state_dict())
