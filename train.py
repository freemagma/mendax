from model import *
from data_control import *
from log import *

brange = None


def run_batch(device, crew, imposter, crew_const=None):
    global brange
    player_ix, crew_views, imposter_views = generate_views(
        B, N, P, I, view_chance=VIEW_CHANCE, sabotage_chance=SABOTAGE_CHANCE
    )
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
        elif crew_const is not None and p == I:
            my_view = crew_views[:, p - I, :, :]
            memory.append(crew.viewer(my_view)[1])  # Train Crew's view pass
        elif crew_const is not None:
            my_view = crew_views[:, p - I, :, :]
            memory.append(crew_const.viewer(my_view)[1])  # Constant crew view pass
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
        elif crew_const is not None and p == I:
            message = crew.comm.get_message(h_0)  # Train crew initial h_0
        elif crew_const is not None:
            message = crew_const.comm.get_message(h_0)  # Constant crew initial h_0
        else:  # Player is crew
            message = crew.comm.get_message(h_0)  # Pass crew initial h_0
        messages[brange, player_ix[:, p], :] = message
    for r in range(ROUNDS):  # Actual communication
        new_messages = torch.zeros((B, P, M)).to(device)  # Next round comm data
        for p in range(P):
            if p < I:  # If player is imposter
                message, memory[p] = imposter.comm(messages.view(B, P * M), memory[p])
                new_messages[brange, player_ix[:, p], :] = message  # Imposter comm data
            elif crew_const is not None and p == I:
                message, memory[p] = crew.comm(messages.view(B, P * M), memory[p])
                new_messages[
                    brange, player_ix[:, p], :
                ] = message  # Train Crew comm data
            elif crew_const is not None:
                message, memory[p] = crew_const.comm(messages.view(B, P * M), memory[p])
                new_messages[
                    brange, player_ix[:, p], :
                ] = message  # Constant crew comm data
            else:  # Player is crew
                message, memory[p] = crew.comm(messages.view(B, P * M), memory[p])
                new_messages[brange, player_ix[:, p], :] = message  # Crew comm data
        messages = new_messages

    # Vote stage
    votes = 0
    for p in range(P):
        if p < I:  # If player is imposter
            votes += imposter.vote(memory[p])  # Imposter votes
        elif crew_const is not None and p == I:
            votes += crew.vote(memory[p])  # Train crew votes
        elif crew_const is not None:
            votes += crew_const.vote(memory[p])  # Constant crew votes
        else:  # Player is crew
            votes += crew.vote(memory[p])  # Crew votes
    votes /= P  # Make values to be [0, 1]
    imposter_votes = votes[brange[:, None], player_ix[:, :I]]  # Extract imposter votes
    return torch.mean(imposter_votes.max(1)[0])  # Mean of max votes for imposter


B = 64
N = 32
H = 256
M = 16
ROUNDS = 8
P = 10
I = 2
EPOCHS = 10
EPOCH_LENGTH = 512
VIEW_CHANCE = 0.4
SABOTAGE_CHANCE = 0.5
TRAIN_CREW_FIRST = False
TRAIN_SINGLE_CREWMATE = True
CONST_CREW_COPY_FREQ = 64
PRINT_FREQ = 64


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    imposter = Agent(device, P, H, M)
    crew = Agent(device, P, H, M)

    crew_const = None
    if TRAIN_SINGLE_CREWMATE:
        crew_const = Agent(device, P, H, M)
        crew_const.requires_grad_(False)

    imposter_optim = torch.optim.Adam(imposter.parameters())
    crew_optim = torch.optim.Adam(crew.parameters())
    optimizer = [imposter_optim, crew_optim]

    running_score = 0
    for e in range(EPOCHS):
        train_crew = (e + TRAIN_CREW_FIRST) % 2
        print_epoch(e, train_crew)

        crew.requires_grad_(train_crew)
        imposter.requires_grad_(not train_crew)
        for b in range(EPOCH_LENGTH):
            optimizer[train_crew].zero_grad()
            crew_score = run_batch(device, crew, imposter, crew_const=crew_const)
            running_score += crew_score
            loss = crew_score * (-1 if train_crew else 1)
            if b % PRINT_FREQ == PRINT_FREQ - 1:
                print(
                    f"    Batch {str(b + 1).zfill(3)}; Crew Score: {running_score / PRINT_FREQ:.3f}"
                )
                running_score = 0
            loss.backward()
            optimizer[train_crew].step()
            if (
                train_crew
                and crew_const is not None
                and b % CONST_CREW_COPY_FREQ == CONST_CREW_COPY_FREQ - 1
            ):
                crew_const.copy_state(crew)

        if train_crew and crew_const is not None:
            crew_const.copy_state(crew)


if __name__ == "__main__":
    main()