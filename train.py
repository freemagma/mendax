from model import *
from view_gen import *


def main():
    N = 4
    H = 32
    C = 5
    R = 3
    players = [0, 0]
    P = len(players)

    viewer = Viewer(P, H)
    comm = Communicator(P, H, C)
    vote = Voter(P, H)

    views = torch.tensor(generate_views(N, players)).float()
    print(views)

    memory = []
    for p in range(P):
        my_view = views[p][:, None, :]
        _, my_memory = viewer(my_view)
        memory.append(my_memory)

    messages = torch.zeros((1, P, C))
    for p in range(P):
        h_0, _ = memory[p]
        messages[:, p, :] = comm.get_message(h_0)
    for r in range(R):
        new_messages = torch.zeros((1, P, C))
        for p in range(P):
            message, memory[p] = comm(messages.view(1, P * C), memory[p])
            new_messages[:, p, :] = message
        messages = new_messages

    votes = 0
    for p in range(P):
        votes += vote(memory[p])
    votes /= P

    print(votes)


if __name__ == "__main__":
    main()