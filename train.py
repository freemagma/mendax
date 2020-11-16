from model import *
from data_control import *
from log import *

brange = None


def run_batch(device, crew, imposter):
    global brange
    player_ix, crew_views, imposter_views = generate_views(B, N, P, I)
    player_ix = torch.tensor(player_ix).long().to(device)
    crew_views = torch.tensor(crew_views).float().to(device)
    imposter_views = torch.tensor(imposter_views).float().to(device)

    # Viewer Stage
    memory = []  # Result from Viewer(view) of each player
    if brange is None:  # Only create brange once
        brange = torch.arange(B).to(device)
    for p in range(P):
        if p < I:  # If player is imposter
            my_view = imposter_views[:, p, :, :]
            memory.append(imposter.viewer(my_view)[1])  # Imposter's view pass
        else:  # Player is crew
            my_view = crew_views[:, p - I, :, :]
            memory.append(crew.viewer(my_view)[1])  # Crew's view pass

    # Communicate Stage
    messages = torch.zeros((B, P, M)).to(device)  # Communication data
    for p in range(P):  # Create initial message
        h_0, _ = memory[p]  # Grab initial memory from player
        message = None
        if p < I:  # If player is imposter
            message = imposter.comm.get_message(h_0)  # Pass imposter initial h_0
        else:  # Player is crew
            message = crew.comm.get_message(h_0)
        messages[brange, player_ix[:, p], :] = message  # Pass crew initial hidden h_0
    for r in range(ROUNDS):  # Actual communication
        new_messages = torch.zeros((B, P, M)).to(device)  # Next round comm data
        for p in range(P):
            if p < I:  # If player is imposter
                message, memory[p] = imposter.comm(messages.view(B, P * M), memory[p])
                new_messages[brange, player_ix[:, p], :] = message  # Imposter comm data
            else:  # Player is crew
                message, memory[p] = crew.comm(messages.view(B, P * M), memory[p])
                new_messages[brange, player_ix[:, p], :] = message  # Crew comm data
        messages = new_messages

    # Vote stage
    votes = 0
    for p in range(P):
        if p < I:  # If player is imposter
            votes += imposter.vote(memory[p])  # Imposter votes
        else:  # Player is crew
            votes += crew.vote(memory[p])  # Player votes
    votes /= P  # Make values to be [0, 1]
    imposter_votes = votes[brange[:, None], player_ix[:, :I]]  # Extract imposter votes
    return torch.mean(imposter_votes.max(1)[0])  # Mean of max votes for imposter


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
        train_crew = (e + 1) % 2
        print_epoch(e, train_crew)

        crew.requires_grad_(train_crew)
        imposter.requires_grad_(not train_crew)
        for b in range(EPOCH_LENGTH):
            optimizer[train_crew].zero_grad()
            crew_score = run_batch(device, crew, imposter)
            running_score += crew_score
            loss = crew_score * (-1 if train_crew else 1)
            if b % 64 == 63:
                print(
                    f"    Batch {str(b + 1).zfill(3)}; Crew Score: {running_score / 64:.3f}"
                )
                running_score = 0
            loss.backward()
            optimizer[train_crew].step()


if __name__ == "__main__":
    main()