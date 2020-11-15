from model import *
from data_control import *
from log import *


def run_batch(device, crew, imposter):
    player_ix, crew_views, imposter_views = generate_views(B, N, P, I)
    player_ix = torch.tensor(player_ix).long().to(device)
    crew_views = torch.tensor(crew_views).float().to(device)
    imposter_views = torch.tensor(imposter_views).float().to(device)

    memory = []
    brange = torch.arange(B).to(device)
    for p in range(P):
        if p < I:
            my_view = imposter_views[:, p, :, :]
            memory.append(imposter.viewer(my_view)[1])
        else:
            my_view = crew_views[:, p - I, :, :]
            memory.append(crew.viewer(my_view)[1])

    messages = torch.zeros((B, P, M)).to(device)
    for p in range(P):
        h_0, _ = memory[p]
        message = None
        if p < I:
            message = imposter.comm.get_message(h_0)
        else:
            message = crew.comm.get_message(h_0)
        messages[brange, player_ix[:, p], :] = message
    for r in range(ROUNDS):
        new_messages = torch.zeros((B, P, M)).to(device)
        for p in range(P):
            if p < I:
                message, memory[p] = imposter.comm(messages.view(B, P * M), memory[p])
                new_messages[brange, player_ix[:, p], :] = message
            else:
                message, memory[p] = crew.comm(messages.view(B, P * M), memory[p])
                new_messages[brange, player_ix[:, p], :] = message
        messages = new_messages

    votes = 0
    for p in range(P):
        if p < I:
            votes += imposter.vote(memory[p])
        else:
            votes += crew.vote(memory[p])
    votes /= P
    imposter_votes = votes[brange[:, None], player_ix[:, :I]]
    return torch.mean(imposter_votes.max(1)[0])


B = 64
N = 16
H = 256
M = 8
ROUNDS = 5
P = 10
I = 2
EPOCHS = 10
EPOCH_LENGTH = 512


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    imposter = Agent(device, P, H, M)
    crew = Agent(device, P, H, M)

    imposter_optim = torch.optim.Adam(imposter.parameters())
    crew_optim = torch.optim.Adam(crew.parameters())
    optimizer = [imposter_optim, crew_optim]

    running_score = 0
    for e in range(EPOCHS):
        train_crew = (e + 0) % 2
        print_epoch(e, train_crew)

        crew.requires_grad_(train_crew)
        imposter.requires_grad_(not train_crew)
        for b in range(EPOCH_LENGTH):
            optimizer[train_crew].zero_grad()
            crew_score = run_batch(device, crew, imposter)
            running_score += crew_score
            loss = crew_score * (-1 if train_crew else 1)
            if b % 64 == 63:
                print(f"    Batch {str(b + 1).zfill(3)}; Crew Score: {crew_score:.3f}")
                running_score = 0
            loss.backward()
            optimizer[train_crew].step()


if __name__ == "__main__":
    main()