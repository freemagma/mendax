from model import *
from data_control import *
from log import *

brange = None


def run_batch(device, crew, imposter, crew_const=None, imposter_const=None):
    global brange
    player_ix, crew_views, imposter_views = generate_views(
        B, N, P, I, view_chance=VIEW_CHANCE, sabotage_chance=SABOTAGE_CHANCE
    )
    player_ix = torch.tensor(player_ix).long().to(device)
    crew_views = torch.tensor(crew_views).float().to(device)
    imposter_views = torch.tensor(imposter_views).float().to(device)

    # Viewer Stage
    memory = []
    if brange is None:  # Only create brange once
        brange = torch.arange(B).to(device)
    for p in range(P):
        # Imposter
        if p < I:
            my_view = imposter_views[:, p, :, :]
            if imposter_const is not None and p != 0:
                memory.append(imposter_const.viewer(my_view)[1])
            else:
                memory.append(imposter.viewer(my_view)[1])
        # Crew
        else:
            my_view = crew_views[:, p - I, :, :]
            if crew_const is not None and p != I:
                memory.append(crew_const.viewer(my_view)[1])
            else:
                memory.append(crew.viewer(my_view)[1])

    # Communicate Stage
    messages = torch.zeros((B, P, M)).to(device)
    for p in range(P):
        h_0, _ = memory[p]
        message = None
        # Imposter
        if p < I:
            if imposter_const is not None and p != 0:
                message = imposter_const.comm.get_message(h_0)
            else:
                message = imposter.comm.get_message(h_0)
        # Crew
        else:
            if crew_const is not None and p != I:
                message = crew_const.comm.get_message(h_0)
            else:
                message = crew.comm.get_message(h_0)
        messages[brange, player_ix[:, p], :] = message
    for r in range(ROUNDS):
        new_messages = torch.zeros((B, P, M)).to(device)
        for p in range(P):
            # Imposter
            if p < I:
                if imposter_const is not None and p != 0:
                    message, memory[p] = imposter_const.comm(
                        messages.view(B, P * M), memory[p]
                    )
                else:
                    message, memory[p] = imposter.comm(
                        messages.view(B, P * M), memory[p]
                    )
            else:
                if crew_const is not None and p != I:
                    message, memory[p] = crew_const.comm(
                        messages.view(B, P * M), memory[p]
                    )
                else:
                    message, memory[p] = crew.comm(messages.view(B, P * M), memory[p])
            new_messages[brange, player_ix[:, p], :] = message
        messages = new_messages

    # Vote stage
    votes = 0
    for p in range(P):
        # Imposter
        if p < I:
            if imposter_const is not None and p != 0:
                votes += imposter_const.vote(memory[p])
            else:
                votes += imposter.vote(memory[p])
        # Crew
        else:
            if crew_const is not None and p != I:
                votes += crew_const.vote(memory[p])
            else:
                votes += crew.vote(memory[p])
    votes /= P
    imposter_votes = votes[brange[:, None], player_ix[:, :I]]

    # Crew score is the mean of max votes in each batch for an imposter
    return torch.mean(imposter_votes.max(1)[0])


B = 128
N = 64
H = 512
M = 20
ROUNDS = 8
P = 10
I = 2
EPOCHS = 10
EPOCH_LENGTH = 1024
VIEW_CHANCE = 0.4
SABOTAGE_CHANCE = 0.7
TRAIN_CREW_FIRST = False
TRAIN_SINGLE_CREWMATE = True
CONST_CREW_COPY_FREQ = 64
TRAIN_SINGLE_IMPOSTER = True
CONST_IMPOSTER_COPY_FREQ = 64
PRINT_FREQ = 64
SAVE_EPOCH_FREQ = 2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    imposter = Agent(device, P, H, M)
    crew = Agent(device, P, H, M)

    crew_const = None
    if TRAIN_SINGLE_CREWMATE:
        crew_const = Agent(device, P, H, M)
        crew_const.requires_grad_(False)

    imposter_const = None
    if TRAIN_SINGLE_IMPOSTER:
        imposter_const = Agent(device, P, H, M)
        imposter_const.requires_grad_(False)

    imposter_optim = torch.optim.Adam(imposter.parameters())
    crew_optim = torch.optim.Adam(crew.parameters())
    optimizer = [imposter_optim, crew_optim]

    running_score = 0
    HALF_EPOCH_LENGTH = EPOCH_LENGTH // 2
    for e in range(EPOCHS):
        train_crew = (e + TRAIN_CREW_FIRST) % 2
        print_epoch(e, train_crew)

        crew.requires_grad_(train_crew)
        imposter.requires_grad_(not train_crew)
        for b in range(EPOCH_LENGTH):
            optimizer[train_crew].zero_grad()
            crew_score = run_batch(
                device,
                crew,
                imposter,
                crew_const=crew_const,
                imposter_const=imposter_const,
            )
            running_score += crew_score
            loss = crew_score * (-1 if train_crew else 1)
            if b % PRINT_FREQ == PRINT_FREQ - 1:
                print(
                    f"    Batch {str(b + 1).zfill(4)}; Crew Score: {running_score / PRINT_FREQ:.3f}"
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
            if (
                not train_crew
                and imposter_const is not None
                and b % CONST_IMPOSTER_COPY_FREQ == CONST_IMPOSTER_COPY_FREQ - 1
            ):
                imposter_const.copy_state(imposter)

        if e % SAVE_EPOCH_FREQ == SAVE_EPOCH_FREQ - 1:
            crew.save_state(f"saves/e{e}_crew")
            imposter.save_state(f"saves/e{e}_imposter")

        if train_crew and crew_const is not None:
            crew_const.copy_state(crew)

        if not train_crew and imposter_const is not None:
            imposter_const.copy_state(imposter)


if __name__ == "__main__":
    main()