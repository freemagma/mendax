import sys
import os
import datetime
import json

from model import *
from data_control import *
from log import *

brange = None


class TrainParams:
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.kwargs = kwargs
        self.kwargs["device"] = self.device.type
        self.BRANGE = torch.arange(self.B).to(self.device)


def run_batch(params, crew, imposter, crew_const=None, imposter_const=None):
    global brange

    player_ix, crew_views, imposter_views = generate_views(
        params.B,
        params.N,
        params.P,
        params.I,
        view_chance=params.VIEW_CHANCE,
        sabotage_chance=params.SABOTAGE_CHANCE,
    )
    player_ix = torch.tensor(player_ix).long().to(params.device)
    crew_views = torch.tensor(crew_views).float().to(params.device)
    imposter_views = torch.tensor(imposter_views).float().to(params.device)

    # Model selector function
    def agent(p):
        if p < params.I and imposter_const is not None and p != 0:
            return imposter_const
        elif p < params.I:
            return imposter
        elif crew_const is not None and p != params.I:
            return crew_const
        else:
            return crew

    # Viewer Stage
    memory = []
    for p in range(params.P):
        my_view = (
            imposter_views[:, p, :, :]
            if p < params.I
            else crew_views[:, p - params.I, :, :]
        )
        memory.append(agent(p).viewer(my_view)[1])

    # Communicate Stage
    messages = torch.zeros((params.B, params.P, params.M)).to(params.device)
    for p in range(params.P):
        h_0, _ = memory[p]
        message = agent(p).comm.get_message(h_0)
        messages[params.BRANGE, player_ix[:, p], :] = message
    for r in range(params.ROUNDS):
        new_messages = torch.zeros((params.B, params.P, params.M)).to(params.device)
        for p in range(params.P):
            message, memory[p] = agent(p).comm(
                messages.view(params.B, params.P * params.M), memory[p]
            )
            new_messages[params.BRANGE, player_ix[:, p], :] = message
        messages = new_messages

    # Vote stage
    votes = 0
    for p in range(params.P):
        votes += agent(p).vote(memory[p])
    votes /= params.P
    imposter_votes = votes[params.BRANGE[:, None], player_ix[:, : params.I]]

    # Crew score is the mean of max votes in each batch for an imposter
    return torch.mean(imposter_votes.max(1)[0])


def train(params, verbose=True, save_dir=None):

    imposter = Agent(params.device, params.P, params.H, params.M)
    crew = Agent(params.device, params.P, params.H, params.M)

    crew_const = None
    if params.TRAIN_SINGLE_CREWMATE:
        crew_const = Agent(params.device, params.P, params.H, params.M)
        crew_const.requires_grad_(False)

    imposter_const = None
    if params.TRAIN_SINGLE_IMPOSTER:
        imposter_const = Agent(params.device, params.P, params.H, params.M)
        imposter_const.requires_grad_(False)

    imposter_optim = torch.optim.Adam(imposter.parameters())
    crew_optim = torch.optim.Adam(crew.parameters())
    optimizer = [imposter_optim, crew_optim]

    if save_dir:
        os.makedirs(save_dir)
        with open(os.path.join(save_dir, "params.json"), "w") as f:
            json.dump(params.kwargs, f, indent=4)

    fs = []
    if verbose:
        fs.append(sys.stdout)
    if save_dir:
        fs.append(open(os.path.join(save_dir, "output.txt"), "w"))

    running_score = 0
    HALF_EPOCH_LENGTH = params.EPOCH_LENGTH // 2
    for e in range(params.EPOCHS):
        train_crew = (e + params.TRAIN_CREW_FIRST) % 2
        print_epoch(e, train_crew, fs)

        crew.requires_grad_(train_crew)
        imposter.requires_grad_(not train_crew)
        for b in range(params.EPOCH_LENGTH):
            optimizer[train_crew].zero_grad()

            crew_score = run_batch(
                params,
                crew,
                imposter,
                crew_const=crew_const,
                imposter_const=imposter_const,
            )
            loss = crew_score * (-1 if train_crew else 1)

            running_score += crew_score
            if b % params.PRINT_FREQ == params.PRINT_FREQ - 1:
                print_batch_score(b, running_score / params.PRINT_FREQ, fs)
                running_score = 0

            loss.backward()
            optimizer[train_crew].step()

            if (
                train_crew
                and crew_const is not None
                and b % params.CONST_CREW_COPY_FREQ == params.CONST_CREW_COPY_FREQ - 1
            ):
                crew_const.copy_state(crew)
            if (
                not train_crew
                and imposter_const is not None
                and b % params.CONST_IMPOSTER_COPY_FREQ
                == params.CONST_IMPOSTER_COPY_FREQ - 1
            ):
                imposter_const.copy_state(imposter)

        if save_dir and e % params.SAVE_EPOCH_FREQ == params.SAVE_EPOCH_FREQ - 1:
            crew.save_state(os.path.join(save_dir, f"e{e}_crew"))
            imposter.save_state(os.path.join(save_dir, f"e{e}_imposter"))

        if train_crew and crew_const is not None:
            crew_const.copy_state(crew)

        if not train_crew and imposter_const is not None:
            imposter_const.copy_state(imposter)

    if save_dir:
        fs[-1].close()


def grid_search(param_dict):
    grid_dir = os.path.join(
        "saves", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_GRID"
    )

    VIEW_GRID = [0.2, 0.4, 0.6, 0.8, 1]
    SAB_GRID = [0, 0.25, 0.5, 0.75, 1]
    for v_chance in VIEW_GRID:
        for s_chance in SAB_GRID:
            param_dict["VIEW_CHANCE"] = v_chance
            param_dict["SABOTAGE_CHANCE"] = s_chance
            params = TrainParams(**param_dict)
            save_dir = os.path.join(grid_dir, f"v{v_chance}_s{s_chance}")
            train(params, save_dir=save_dir, verbose=True)


def main():
    param_dict = {
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "B": 64,
        "N": 12,
        "H": 256,
        "M": 20,
        "ROUNDS": 5,
        "P": 10,
        "I": 2,
        "EPOCHS": 6,
        "EPOCH_LENGTH": 1024,
        "VIEW_CHANCE": 0.4,
        "SABOTAGE_CHANCE": 0.0,
        "TRAIN_CREW_FIRST": False,
        "TRAIN_SINGLE_CREWMATE": True,
        "CONST_CREW_COPY_FREQ": 32,
        "TRAIN_SINGLE_IMPOSTER": True,
        "CONST_IMPOSTER_COPY_FREQ": 32,
        "PRINT_FREQ": 64,
        "SAVE_EPOCH_FREQ": 2,
    }
    # run the grid search
    # grid_search(param_dict)
    # just train once. dont save.
    train(TrainParams(**param_dict), verbose=True)


if __name__ == "__main__":
    main()