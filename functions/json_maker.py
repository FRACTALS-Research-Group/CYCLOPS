import json
import random
from pathlib import Path

def generate_json_files(pos, i, seq, chain_id, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_data = {
        "name": f"seq_{pos}_{i}",
        "sequences": [
            {
                "protein": {
                    "id": [chain_id],
                    "sequence": seq
                }
            }
        ],
        "modelSeeds": [1],
        "dialect": "alphafold3",
        "version": 1
    }
    
    json_path = output_dir / f"seq_{pos}_{i}.json"
    with open(json_path, "w") as file:
        json.dump(json_data, file, indent=2)
    return json_path

# Esempio di utilizzo:
# generate_json_files(ref_seq, keep_pos, amino_acids, start_iteration, "output_json_dir")
