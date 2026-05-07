import json
import os
from pathlib import Path
from typing import Any, Dict


def generate_json_files(pos: str, i: int, mutate_seq: str, output_dir: str | os.PathLike,
    logger) -> str:
    """
        Generate the AlphaFold3 input JSON file for a given iteration/mutation.

        This function keeps the same intent as your existing pipeline:
        it creates a JSON file under output_dir and returns its path.

        Parameters
        ----------
        pos : str
            Mutation position label (e.g. "3_7").
        i : int
            Iteration index.
        mutate_seq : str
            Mutated sequence.
        output_dir : str | os.PathLike
            Directory where the JSON file will be created.
        logger : logging.Logger
            Mandatory logger.

        Returns
        -------
        str
            Path to the generated JSON file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"seq_{pos}_{i}.json"


    payload: Dict[str, Any] = {
        "name": f"seq_{pos}_{i}",
        "sequence": mutate_seq,
        "metadata": {
            "iteration": int(i),
            "pos": str(pos),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Generated AF3 JSON: %s", json_path)
    return str(json_path)

