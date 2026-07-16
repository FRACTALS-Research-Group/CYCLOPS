import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

from functions.pdb_converters import pdb2pdbqt_lig, pdb2pdbqt_rece

def setup_logger(log_file: str | Path) -> logging.Logger:
    """
    Sets up a logger that writes to both a rotating file and the console.
    
    Parameters
    ----------
    log_file : str or Path
        The path to the log file. The logger will create this file and rotate it when it exceeds 5 MB, keeping up to 5 backup files.
    
    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    logger = logging.getLogger("MC")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers on repeated calls
    if getattr(logger, "_configured", False):
        return logger

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(str(log_file), maxBytes=5_000_000, backupCount=5)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(fmt)

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    logger.addHandler(handler)
    logger.addHandler(console)

    logger._configured = True 
    return logger



def load_step0_results(step0_results_path: Path, logger: Any) -> Dict[str, float]:
    """
        Load the reference docking scores saved by step_0.py.

        Parameters
        ----------
        step0_results_path : Path
            Path to the JSON file produced by step_0.py.
        logger : Any
            Logger instance.

        Returns
        -------
        Dict[str, float]
            Dictionary with keys ``score_pre_simulation`` and ``score_post_simulation``.

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist (step_0.py has not been run yet).
    """
    step0_results_path = Path(step0_results_path)
    if not step0_results_path.exists():
        raise FileNotFoundError(
            f"step_0 results file not found: {step0_results_path}. "
            "Run step_0.py first and point --results_path to the same location "
            "used by --step0_results in main.py."
        )
    with open(step0_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded step-0 results from %s: %s", step0_results_path, data)
    return data


def setup_directories(directories):
    """
        Creates the specified directories if they do not already exist.
        
        Parameters
        ----------
        directories : iterable of str or Path
            An iterable of directory paths to create. Each path can be a string or a Path object.
    """
    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)

def init_scores_file(scores_file: Path, ref_seq: str, step0_results_path: Path, logger: Any) -> None:
    """
        Initialize docking score CSV if missing.

        Parameters
        ----------
        scores_file : Path
            Path to the CSV file where docking scores will be recorded. If the file does not exist, it will be created with a header and an initial entry for the reference sequence.
        ref_seq : str
            The reference sequence to be recorded in the initial entry of the scores file.
        step0_results_path : Path
            Path to the JSON file produced by step_0.py containing the reference docking scores.
    """
    if not scores_file.exists():
        results = load_step0_results(step0_results_path, logger)
        score_pre = results["score_pre_simulation"]
        score_post = results["score_post_simulation"]
        scores_file.parent.mkdir(parents=True, exist_ok=True)
        with open(scores_file, "w", encoding="utf-8") as f:
            f.write("iteration,position,sequence,score_pre_simulation,score_post_simulation,delta_e\n")
            f.write(f"0,ref,{ref_seq},{score_pre},{score_post},0.00\n")
        logger.info("Initialized scores file: %s", scores_file)

def log_time(start: float, end: float, i: int, logger: Any) -> None:
    """
        Log the elapsed time for an iteration.
        
        Parameters
        ----------
        start : float
            The start time of the iteration, typically obtained from time.time() at the beginning of the iteration.
        end : float
            The end time of the iteration, typically obtained from time.time() at the end of the iteration.
        i : int
            The current iteration number, used for logging purposes.
        logger : Any
            The logger instance to use for logging the elapsed time information.
    """
    elapsed = end - start
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    logger.info("Iteration %d completed in %02d:%02d:%02d", int(i), int(hrs), int(mins), int(secs))


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


def setup_paths(args: argparse.Namespace) -> Tuple[Dict[str, Path], Path]:
    """
        Sets up the directory paths based on the provided base directory and ensures that all necessary directories exist.
        
        Parameters
        ----------
        args : argparse.Namespace
            The parsed command-line arguments containing the base directory path and other configuration options.
        
        Returns
        -------
        Tuple[Dict[str, Path], Path]
            A tuple containing a dictionary of directory paths and the base directory path.
    """
    base = Path(args.base_dir)
    paths: Dict[str, Path] = {
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
        "STEP0_RESULTS": base / "step0_results.json",
        "GLOBAL_RESULTS": base / "global_results",
        "TOP_10": base / "results_top10",
    }
    setup_directories(v for k, v in paths.items() if k != "STEP0_RESULTS")
    return paths, base


def setup_refs(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    base: Path,
    logger: Any,
) -> Tuple[str, str]:
    """
        Prepare receptor/ligand pdbqt references and return their pdbqt paths as strings.

        Parameters
        ----------
        args : argparse.Namespace
            The parsed command-line arguments containing the input file names and other configuration options.
        paths : Dict[str, Path]
            A dictionary containing the directory paths for inputs and outputs.
        base : Path
            The base directory path used for constructing the full paths to the input and output files.
        logger : Any
            The logger instance to use for logging information about the reference preparation process.
        
        Returns
        -------
        Tuple[str, str]
            A tuple containing the file paths to the prepared receptor and ligand pdbqt files as strings.
    """
    ref_receptor_pdb = paths["REF_INPUT_FOLDER"] / f"{args.input_rece_file}"
    ref_ligand_pdb = paths["REF_INPUT_FOLDER"] / f"{args.ref_lig_file}"

    pdb2pdbqt_rece(str(ref_receptor_pdb), str(paths["REF_OUTPUT_FOLDER"]), str(base), conversion="*.pdb", logger=logger)
    pdb2pdbqt_lig(str(ref_ligand_pdb).replace(".pdb", ""), str(paths["REF_OUTPUT_FOLDER"]), str(base), logger=logger)

    receptor_path = paths["REF_OUTPUT_FOLDER"] / f"{ref_receptor_pdb.stem}.pdbqt"
    ref_ligand_path = paths["REF_OUTPUT_FOLDER"] / f"{ref_ligand_pdb.stem}.pdbqt"

    logger.info("Reference receptor pdbqt: %s", receptor_path)
    logger.info("Reference ligand pdbqt: %s", ref_ligand_path)

    return str(receptor_path), str(ref_ligand_path)
