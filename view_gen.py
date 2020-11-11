import torch
import numpy as np


def generate_views(N, players, view_chance=0.4, sabotage_chance=0.8):
    """
    INPUTS
    N: number of events
    players: list [0, 0, 0, 1] where 1s indicate imposters

    OUTPUT
    P x N x (P+1) matrix indicating the view of each player,
    where each player experiences N events. During each event they
    see some set of coplayers, and sometimes experience a sabotage.
    """
    P = len(players)
    player_ix = [i for i, v in enumerate(players) if not v]

    # generate N, P, P symmetric matrix
    # events, viewee, viewer
    groups = np.random.rand(N, P, P)
    groups = (groups + groups.transpose(0, 2, 1)) / 2
    groups = groups < view_chance
    groups[:, np.arange(P), np.arange(P)] = True

    sabotages = np.random.rand(N, P) < sabotage_chance
    sabotages[:, player_ix] = False

    # (N, 1, P) matrix representing who sees a sabotage
    see_sabotage = np.any(np.logical_and(groups, sabotages[:, :, None]), axis=1)[
        :, None, :
    ]

    views = np.logical_and(groups, np.logical_not(sabotages)[:, :, None])
    views[:, np.arange(P), np.arange(P)] = True
    views = np.concatenate((views, see_sabotage), axis=1)

    return views.transpose(2, 0, 1)


def test():
    v = generate_views(10, [0, 0, 0, 0, 1])
    print(v.shape)
    print(v)


if __name__ == "__main__":
    test()
