import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import argparse
from functions.pdb_converters import pdb2pdbqt_lig, pdb2pdbqt_rece

def setup_logger(log_file):
    logger = logging.getLogger("MC")
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def setup_directories(directories):
    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)

def init_scores_file(scores_file, ref_seq):
    if not scores_file.exists():
        with open(scores_file, "w") as f:
            f.write("iteration,position,sequence,score_pre_simulation,score_post_simulation,delta_e\n")
            f.write(f"0,ref,{ref_seq},-6.87,-7.66,0.00\n")

def log_time(start, end, i, logger):
    elapsed = end - start
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    logger.info(f"Iteration {i} completed in {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}")

def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--base_dir", required=True, help="Base directory path")  # ALLIANCE
    parser.add_argument("--input_rece_file", required=True, help="Receptor file name")  # rece_1AKJ
    parser.add_argument("--ref_lig_file", required=True, help="Reference ligand file name")  # lig_1AKJ OR pocket1
    parser.add_argument("--out_base", type=str, default="system", help="Name of the outputs for MD simulation")
    parser.add_argument("--steps", type=int, default=5000000, help="Number of simulation steps") # 5000000
    parser.add_argument("--step_size", type=float, default=0.002, help="Size of each simulation step")
    parser.add_argument("--temperature", type=float, default=2, help="Metropolis temperature")
    parser.add_argument("--ref_seq", type=str, default="CAAAAAAAAAAAC", help="Reference sequence")  # E.g., CAADQTQDTEAAC
    parser.add_argument("--keep_pos", type=int, nargs='+', default=[0, 1], help="Positions to keep unchanged")  # E.g., 0 5 6 7 12
    parser.add_argument("--iter", type=int, default=201, help="Number of iterations of the loop")
    parser.add_argument("--n_mut", type=int, default=1, help="Number of mutations attempted")
    return parser.parse_args()

def setup_paths(args):
    base = Path(args.base_dir)
    paths = {
        "REF_INPUT_FOLDER": base / "inputs" / "ref" / "pdb",
        "REF_OUTPUT_FOLDER": base / "inputs" / "ref" / "pdbqt",
        "LIG_INPUT_FOLDER": base / "inputs" / "ligands" / "pdb",
        "LIG_OUTPUT_FOLDER": base / "inputs" / "ligands" / "pdbqt",
        "JSON_DIR": base / "inputs" / "ligands" / "json",
        "DOCKING_IN_DIR": base / "docking" / "inputs",
        "MINIMIZED_OUTPUT_PATH": base / "docking" / "minimized_outputs",
        "DOCKING_OUT_PATH": base / "docking" / "outputs",
        "OUTPUT_DIR": base / "output",
        "OUTPUT_PDB": base / "output" / "complexes",
        "DOCKING_SCORES": base / "output" / "analysis",
        "CHECKPOINT_DIR": base / "checkpoints",
        "GLOBAL_RESULTS": base / "global_results",
        "TOP_10": base / "results_top10"
    }
    setup_directories(paths.values())
    return paths, base


def setup_refs(args, paths, base):
    ref_receptor_pdb = paths["REF_INPUT_FOLDER"] / f"{args.input_rece_file}"
    ref_ligand_pdb = paths["REF_INPUT_FOLDER"] / f"{args.ref_lig_file}"
    pdb2pdbqt_rece(ref_receptor_pdb, paths["REF_OUTPUT_FOLDER"], base)
    pdb2pdbqt_lig(ref_ligand_pdb, paths["REF_OUTPUT_FOLDER"], base)
    receptor_path = paths["REF_OUTPUT_FOLDER"] / f"{ref_receptor_pdb.stem}.pdbqt"
    ref_ligand_path = paths["REF_OUTPUT_FOLDER"] / f"{ref_ligand_pdb.stem}.pdbqt"
    return str(receptor_path), str(ref_ligand_path)

args = parse_args()
paths, base = setup_paths(args)
setup_refs(args, paths, base)