def print_epoch(e, train_crew):
    out1 = f"│ Epoch {e} (Training " + ("Crew" if train_crew else "Imposters") + ") │"
    print("┌" + "─" * (len(out1) - 2) + "┐")
    print(out1)
    print("└" + "─" * (len(out1) - 2) + "┘")