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
        view: N x B x (P+1)

        OUTPUTS
        """
        B = view.shape[1]
        h0 = self.h0.repeat(1, B, 1)
        c0 = self.c0.repeat(1, B, 1)
        print(view.shape, h0.shape, c0.shape)
        return self.lstm(view, (h0, c0))


class Communicator(nn.Module):
    def __init__(self, P, H, C):
        super(Communicator, self).__init__()
        self.fc1 = nn.Linear(H, C)
        self.fc2 = nn.Linear(C, C)
        self.lstm = nn.LSTM(P * C, H)

    def forward(self, messages, hc_t):
        """
        INPUTS
        messages: N x (P * C)
        hc_t: tuple (h_t, c_t)

        OUTPUTS
        tuple (message, hc_t1)
        message: N x C communication to other agents
        hc_t1: tuple (h_t1, c_t1)
        """
        print(messages[None, :, :].shape, hc_t[0].shape, hc_t[1].shape)
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
        self.softmax = nn.Softmax(1)

    def forward(self, hc_n):
        x = torch.cat(hc_n, dim=2)
        x = torch.relu(self.fc1(x))
        return self.softmax(self.fc2(x))