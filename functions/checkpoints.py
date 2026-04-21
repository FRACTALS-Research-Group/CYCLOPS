from pathlib import Path
import pickle

def save_checkpoint(
    i, 
    checkpoint_file_base, 
    ref_seq, 
    old_pos, 
    old_i, 
    old_docking_score,
    temperature,
    interval=10
):
    """
    Saves a checkpoint periodically based on the given interval.

    Args:
        i (int): The current iteration.
        checkpoint_file_base (Path): Base directory or file path for checkpoints.
        ref_seq (object): Reference sequence to save.
        old_pos (object): Old position data to save.
        old_i (object): Previous index data to save.
        interval (int): Interval at which to save checkpoints (default: 10).
    """
    if i % interval == 0:
        # Generate the checkpoint filename
        checkpoint_filename = checkpoint_file_base / f"checkpoint_{i}.pkl"
        
        # Prepare the checkpoint data
        checkpoint_data = {
            'ref_seq': ref_seq,
            'old_pos': old_pos,
            'old_i': old_i,
            'iteration': i + 1,
            'old_docking_score': old_docking_score,
            "temperature": temperature
        }
        
        # Save the checkpoint using pickle
        with open(checkpoint_filename, 'wb') as file:
            pickle.dump(checkpoint_data, file)
        
        print(f"Checkpoint saved: {checkpoint_filename}")

from pathlib import Path
import pickle
import re

def load_latest_checkpoint(
    checkpoint_dir,
    ref_seq,
    old_pos,
    old_i,
    old_docking_score,
    T
):
    checkpoint_dir = Path(checkpoint_dir)

    # Cerca file tipo checkpoint_XX.pkl
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pkl"))

    if not checkpoint_files:
        print("⚠️  Nessun checkpoint trovato, uso valori iniziali.")
        return ref_seq, old_pos, old_i, old_docking_score, 0, T

    # Ordina per indice numerico
    def extract_number(path):
        m = re.search(r"checkpoint_(\d+)\.pkl", path.name)
        return int(m.group(1)) if m else -1

    checkpoint_files.sort(key=extract_number)

    latest = checkpoint_files[-1]
    print(f"🔄 Carico il checkpoint: {latest}")

    with open(latest, "rb") as f:
        data = pickle.load(f)

    # Estrae solo i campi che ti servono
    return (
        data.get("ref_seq", ref_seq),
        data.get("old_pos", old_pos),
        data.get("old_i", old_i),
        data.get("old_docking_score", old_docking_score),
        data.get("iteration", old_i),  # nemmeno tu usavi "iteration" come campo
        data.get("T", T)
    )




# import pickle

# with open("checkpoint_20.pkl", "rb") as f:
#     data = pickle.load(f)

# print(type(data))
# print(data)
