def print_epoch(e, train_crew, fs):
    out1 = f"│ Epoch {e} (Training " + ("Crew" if train_crew else "Imposters") + ") │"
    for f in fs:
        print("┌" + "─" * (len(out1) - 2) + "┐", file=f)
        print(out1, file=f)
        print("└" + "─" * (len(out1) - 2) + "┘", file=f)


def print_batch_score(b, score, fs):
    for f in fs:
        print(
            f"    Batch {str(b + 1).zfill(4)}; Crew Score: {score:.3f}",
            file=f,
        )