from pathlib import Path
from typing import Any, Optional, Tuple
import re
import pickle

import pickle

def save_checkpoint(i: int, checkpoint_file_base: Path | str, ref_seq: str, old_pos: Optional[str],
    old_i: Optional[int], old_docking_score: float, temperature: float, logger, interval: int = 10) -> None:
    """
        Saves a checkpoint of the current state of the optimization process.
        
        Parameters
        ----------
        i : int
            Current iteration number.
        checkpoint_file_base : Path | str
            Base directory or file path for checkpoints.
        ref_seq : str
            Reference sequence to save in the checkpoint.
        old_pos : Optional[str]
            Previous position data to save in the checkpoint.
        old_i : Optional[int]
            Previous index data to save in the checkpoint.
        old_docking_score : float
            Previous docking score to save in the checkpoint.
        temperature : float
            Current temperature to save in the checkpoint.
        logger : logging.Logger
            Logger for logging information and errors.
        interval : int, optional
            Interval at which to save checkpoints (default is 10).
        
        Raises
        ------
        ValueError
            If the interval is not greater than 0.
    """
    checkpoint_file_base = Path(checkpoint_file_base)
    checkpoint_file_base.mkdir(parents=True, exist_ok=True)

    if interval <= 0:
        raise ValueError(f"interval must be > 0 (got {interval})")

    if i % interval != 0:
        return

    checkpoint_filename = checkpoint_file_base / f"checkpoint_{i}.pkl"

    checkpoint_data: dict[str, Any] = {
        "ref_seq": ref_seq,
        "old_pos": old_pos,
        "old_i": old_i,
        "iteration": i + 1,
        "old_docking_score": float(old_docking_score),
        "temperature": float(temperature),
    }

    with open(checkpoint_filename, "wb") as file:
        pickle.dump(checkpoint_data, file)

    logger.info("Checkpoint saved: %s", checkpoint_filename)

def load_latest_checkpoint(checkpoint_dir: Path | str, ref_seq: str, old_pos: Optional[str],
        old_i: Optional[int], old_docking_score: float, T: float, logger
    ) -> Tuple[str, Optional[str], Optional[int], float, int, float]:
    """
        Loads the latest checkpoint from the specified directory.
        
        Parameters
        ----------
        checkpoint_dir : Path | str
            Directory where checkpoints are stored.
        ref_seq : str
            Default reference sequence to return if no checkpoint is found.
        old_pos : Optional[str]
            Default previous position data to return if no checkpoint is found.
        old_i : Optional[int]
            Default previous index data to return if no checkpoint is found.
        old_docking_score : float
            Default previous docking score to return if no checkpoint is found.
        T : float
            Default temperature to return if no checkpoint is found.
        logger : logging.Logger
            Logger for logging information and errors.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pkl"))
    if not checkpoint_files:
        logger.warning("No checkpoint found in %s. Using initial values.", checkpoint_dir)
        return ref_seq, old_pos, old_i, float(old_docking_score), 0, float(T)

    def extract_number(path: Path) -> int:
        """
            Extracts the iteration number from a checkpoint file name.
            
            Parameters
            ----------
            path : Path
                Path to the checkpoint file.
            
            Returns
            -------
            int
                The extracted iteration number, or -1 if the pattern does not match.
        """
        m = re.search(r"checkpoint_(\d+)\.pkl", path.name)
        return int(m.group(1)) if m else -1

    checkpoint_files.sort(key=extract_number)
    latest = checkpoint_files[-1]
    logger.info("Loading checkpoint: %s", latest)

    with open(latest, "rb") as f:
        data: dict[str, Any] = pickle.load(f)

    loaded_T = data.get("temperature", data.get("T", T))

    return (
        str(data.get("ref_seq", ref_seq)),
        data.get("old_pos", old_pos),
        data.get("old_i", old_i),
        float(data.get("old_docking_score", old_docking_score)),
        int(data.get("iteration", 0)),
        float(loaded_T),
    )

