#
# Copyright (c) 2026 FRACTALS Research Group
# Visit the Research Group website for more information: https://fractals.group/
#


"""
Monte Carlo optimization pipeline for peptide-protein binding.

This script orchestrates:
    - sequence mutation
    - AlphaFold3 prediction
    - docking
    - molecular dynamics simulations
    - post-processing and analysis
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng, Generator

from functions.setup import (
    setup_logger,
    setup_paths,
    setup_refs,
    init_scores_file,
    parse_args,
    log_time,
)
from functions.mutations import random_n_mutation
from functions.vinadock import vina_dock
from functions.simulate_complex import simulate_complex
from functions.analysis import run_analysis, reimage_trajectory
from functions.clustering import get_most_probable_conformation
from functions.af3 import alphafold3
from functions.json_maker import generate_json_files
from functions.pdb_process import split_pdb_by_chain
from functions.checkpoints import save_checkpoint, load_latest_checkpoint
from functions.pdb_converters import cif2pdb, pdb2pdbqt_rece, pdb2pdbqt_lig, pdbqt2pdb


# -------------------
# Small utilities
# -------------------

AMINO_ACIDS: Sequence[str] = tuple("ARNDQEGHILKMFPSTWYV")

def _ensure_dir(path: os.PathLike | str) -> None:
    """
        Create a directory if it doesn't exist.
        
        Parameters
        ----------
        path : os.PathLike or str
            Directory path to create.
        
        Notes
        -----
        - Uses parents=True to create any necessary parent directories.
        - Uses exist_ok=True to avoid error if the directory already exists.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    """
        Best-effort float conversion with fallback.
        
        Parameters
        ----------
        x : Any
            Input to convert to float.
        default : float, optional
            Value to return if conversion fails (default: NaN).
    """
    try:
        return float(x)
    except Exception:
        return default


def _get_seed_from_args(args: Any) -> Optional[int]:
    """
        Extracts the seed value from the command-line arguments if it exists.
        
        Parameters
        ----------
        args : Any
            Parsed command-line arguments, expected to have a 'seed' attribute.
        Returns
        -------
        Optional[int]
            The seed value if it exists and is an integer, otherwise None.
    """
    return getattr(args, "seed", None)


@dataclass(frozen=True)
class AcceptanceResult:
    """
        Data class to hold the result of the accept/reject decision in the Metropolis criterion.
    """
    ref_seq: str
    old_docking_score: float
    old_pos: Optional[str]
    old_i: Optional[int]
    accepted: bool
    delta_e: float
    p_acc: float
    x: float


# -------------------
# Workflow functions
# -------------------

def propose_mutation(
    ref_seq: str,
    keep_pos: Iterable[int],
    amino_acids: Sequence[str],
    n_mut: int,
) -> Tuple[str, str]:
    """
        Propose a new mutated sequence by mutating n_mut positions not in keep_pos.
        
        Parameters
        ----------
        ref_seq : str
            The reference sequence to mutate.
        keep_pos : Iterable[int]
            Positions to keep unchanged (0-based indices).
        amino_acids : Sequence[str]
            List of possible amino acids for mutation.
        n_mut : int
            Number of positions to mutate.

        Returns
        -------
        mutate_seq : str
            Mutated sequence.
        pos : str
            Position string, e.g. "3_7_9".
    """
    keep_pos_set = set(keep_pos)
    valid_indices = [j for j in range(len(ref_seq)) if j not in keep_pos_set]
    mutate_seq, pos_list = random_n_mutation(ref_seq, valid_indices, amino_acids, n=n_mut)
    return mutate_seq, "_".join(map(str, pos_list))


def run_docking(
    i: int,
    pos: str,
    mutate_seq: str,
    JSON_DIR: os.PathLike | str,
    LIG_INPUT_FOLDER: os.PathLike | str,
    LIG_OUTPUT_FOLDER: os.PathLike | str,
    MINIMIZED_OUTPUT_PATH: os.PathLike | str,
    DOCKING_OUT_PATH: os.PathLike | str,
    receptor_path: os.PathLike | str,
    ref_ligand_path: os.PathLike | str,
    base_dir: os.PathLike | str,
    logger,
) -> float:
    """
        Run AF3 structure prediction and docking pre-simulation.
        
        Parameters
        ----------
        i : int
            Current iteration number.
        pos : str
            Mutation position string, e.g. "3_7_9".
        mutate_seq : str
            The mutated sequence to predict and dock.
        JSON_DIR : os.PathLike or str
            Directory to store generated JSON files for AF3.
        LIG_INPUT_FOLDER : os.PathLike or str
            Directory to store AF3 output PDB files for ligands.
        LIG_OUTPUT_FOLDER : os.PathLike or str
            Directory to store converted PDBQT files for ligands.
        MINIMIZED_OUTPUT_PATH : os.PathLike or str
            Directory to store minimized PDBQT files for docking.
        DOCKING_OUT_PATH : os.PathLike or str
            Directory to store docking output PDBQT files.
        receptor_path : os.PathLike or str
            Path to the receptor PDB file.
        ref_ligand_path : os.PathLike or str
            Path to the reference ligand PDB file.
        base_dir : os.PathLike or str
            Base directory for the project, used for relative paths in conversions.
        logger : logging.Logger
            Logger for logging information and errors.

        Returns
        -------
        best_score : float
            Best docking score from the pre-simulation docking step (kcal/mol).
    """
    # Generate AF3 input JSON + run AF3
    json_path = generate_json_files(pos, i, mutate_seq, output_dir=str(JSON_DIR))
    logger.info("Generated JSON for AF3: %s", json_path)

    alphafold3(json_path, output_dir=str(LIG_INPUT_FOLDER), logger=logger)
    logger.info("AlphaFold3 completed for iter=%s pos=%s", i, pos)

    # Convert CIF -> PDB
    cif_in = f"{LIG_INPUT_FOLDER}/seq_{pos}_{i}/seq_{pos}_{i}_model.cif"
    pdb_out = f"{LIG_INPUT_FOLDER}/seq_{pos}_{i}/seq_{pos}_{i}.pdb"
    cif2pdb(cif_in, pdb_out, logger)

    # Convert ligand PDB -> PDBQT
    pdb2pdbqt_lig(
        f"{LIG_INPUT_FOLDER}/seq_{pos}_{i}/seq_{pos}_{i}",
        f"{LIG_OUTPUT_FOLDER}/seq_{pos}_{i}",
        str(base_dir),
        conversion=f"seq_{pos}_{i}.pdb",
        logger=logger
    )

    # Docking pre-simulation using AutoDock Vina
    best_score, _ = vina_dock(
        str(receptor_path),
        str(ref_ligand_path),
        f"{LIG_OUTPUT_FOLDER}/seq_{pos}_{i}/seq_{pos}_{i}.pdbqt",
        f"{MINIMIZED_OUTPUT_PATH}/min_{pos}_{i}.pdbqt",
        f"{DOCKING_OUT_PATH}/out_{pos}_{i}.pdbqt",
        box_size=[20, 20, 20],
        exhaustiveness=32,
        n_poses=10,
        logger=logger
    )

    # Convert docked poses to PDB for MD input
    pdbqt2pdb(str(DOCKING_OUT_PATH), f"{DOCKING_OUT_PATH}/pdb", str(base_dir), pos, i, logger=logger)

    best_score_f = _safe_float(best_score)
    logger.info("Docking pre-simulation score=%.3f (iter=%d pos=%s)", best_score_f, i, pos)
    return best_score_f


def run_full_simulation(
    i: int,
    pos: str,
    receptor_path: os.PathLike | str,
    OUTPUT_PDB: os.PathLike | str,
    DOCKING_IN_DIR: os.PathLike | str,
    MINIMIZED_OUTPUT_PATH: os.PathLike | str,
    DOCKING_OUT_PATH: os.PathLike | str,
    args: Any,
    logger,
    base_dir: os.PathLike | str,
) -> float:
    """
        Run MD simulation, analysis, representative-frame extraction, and post-MD docking.
        
        Parameters
        ----------
        i : int
            Current iteration number.
        pos : str
            Mutation position string, e.g. "3_7_9".
        receptor_path : os.PathLike or str
            Path to the receptor PDB file.
        OUTPUT_PDB : os.PathLike or str
            Directory to store output PDB files from the simulation.
        DOCKING_IN_DIR : os.PathLike or str
            Directory to store input files for post-MD docking.
        MINIMIZED_OUTPUT_PATH : os.PathLike or str
            Directory to store minimized PDBQT files for docking.
        DOCKING_OUT_PATH : os.PathLike or str
            Directory to store docking output PDBQT files.
        args : Any
            Parsed command-line arguments, expected to have attributes like steps, out_base, etc.
        logger : logging.Logger
            Logger for logging information.
        base_dir : os.PathLike or str
            Base directory for the project, used for relative paths in conversions.

        Returns
        -------
        best_score_f : float
            Best docking score from the post-simulation docking step (kcal/mol).
    """
    seq_dir = Path(OUTPUT_PDB) / f"seq_{pos}_{i}"
    _ensure_dir(seq_dir)

    ligand_pdb = f"{DOCKING_OUT_PATH}/pdb/out_{pos}_{i}.pdb"

    # --- MD simulation ---
    try:
        simulate_complex(
            protein_pdb_path=str(receptor_path),
            ligand_pdb_path=str(ligand_pdb),
            base_out=str(seq_dir),
            output_base=f"{seq_dir}/{args.out_base}_{pos}_{i}",
            steps=args.steps,
            solvate=True,
            annealing=False,
            restrain_protein=False,
            T_restrain=400,
            temp_start=1000,
            temp_end=310,
        )
        logger.info("MD simulation completed (iter=%d pos=%s)", i, pos)
    except Exception:
        logger.exception("MD simulation failed (iter=%d pos=%s)", i, pos)
        raise

    # --- Post-MD analysis ---
    run_analysis(args.base_dir, args.out_base, pos, i, logger=logger)
    reimage_trajectory(args.base_dir, args.out_base, pos, i, logger=logger)

    dock_dir = Path(DOCKING_IN_DIR) / f"seq_{pos}_{i}"
    _ensure_dir(dock_dir)

    # --- Representative frame & chain splitting ---
    reimaged_traj = f"{seq_dir}/{args.out_base}_{pos}_{i}_traj_reimaged.pdb"

    representative_frame, _, _ = get_most_probable_conformation(
        reimaged_traj,
        pos,
        i,
        output_path=str(seq_dir),
        subsample_rate=20,
        logger=logger
    )
    logger.info(
        "Representative frame selected: %s (iter=%d pos=%s)",
        representative_frame, i, pos
    )

    split_pdb_by_chain(
        f"{seq_dir}/rep_mut_{pos}_{i}.pdb",
        f"{dock_dir}/rep_rece_{pos}_{i}.pdb",
        f"{dock_dir}/rep_lig_{pos}_{i}.pdb",
        logger=logger
    )

    # Convert representative receptor/ligand for docking
    pdb2pdbqt_rece(str(dock_dir), str(dock_dir), str(base_dir), conversion=f"rep_rece_{pos}_{i}.pdb", logger=logger)
    pdb2pdbqt_lig(
        f"{dock_dir}/rep_lig_{pos}_{i}",
        str(dock_dir),
        str(base_dir),
        conversion=f"rep_lig_{pos}_{i}.pdb",
        logger=logger
    )

    # --- Docking post-MD ---
    best_score, _ = vina_dock(
        f"{dock_dir}/rep_rece_{pos}_{i}.pdbqt",
        f"{dock_dir}/rep_lig_{pos}_{i}.pdbqt",
        f"{dock_dir}/rep_lig_{pos}_{i}.pdbqt",
        f"{MINIMIZED_OUTPUT_PATH}/min_system_{pos}_{i}.pdbqt",
        f"{DOCKING_OUT_PATH}/out_system_{pos}_{i}.pdbqt",
        dock=False,
        box_size=[20, 20, 20],
        exhaustiveness=32,
        n_poses=10,
        logger=logger
    )

    best_score_f = _safe_float(best_score)
    logger.info("Docking post-simulation score=%.3f (iter=%d pos=%s)", best_score_f, i, pos)
    return best_score_f


def accept_or_reject(
    *,
    i: int,
    pos: str,
    mutate_seq: str,
    best_docking_score_pre: float,
    best_docking_score: float,
    old_docking_score_pre: float,
    old_docking_score: float,
    ref_seq: str,
    T: float,
    rng: Generator,
    xrange: Tuple[float, float],
    DOCKING_SCORES_FILE: os.PathLike | str,
    logger,
) -> AcceptanceResult:
    """
    Metropolis criterion using post-simulation score as the energy proxy.
    
    Parameters
    ----------
    i : int
        Current iteration number.
    pos : str
        Mutation position string, e.g. "3_7_9".
    mutate_seq : str
        The mutated sequence proposed in this iteration.
    best_docking_score_pre : float
        The docking score before the MD simulation for the proposed mutation.
    best_docking_score : float
        The docking score after the MD simulation for the proposed mutation.
    old_docking_score_pre : float
        The docking score before the MD simulation for the current reference sequence.
    old_docking_score : float
        The docking score after the MD simulation for the current reference sequence.
    ref_seq : str
        The current reference sequence before mutation.
    T : float
        Current temperature for the Metropolis criterion.
    rng : Generator
        Random number generator for stochastic acceptance.
    xrange : Tuple[float, float]
        Range for uniform random number generation (e.g., (0.0, 1.0)).
    DOCKING_SCORES_FILE : os.PathLike or str
        Path to the CSV file where docking scores and mutation information will be logged.
    logger : logging.Logger
        Logger for logging information.   

    Notes
    -----
    - Accept if delta_e < 0 (i.e., new score is better), otherwise accept with probability p_acc = exp(-delta_e / T).

    Returns
    -------
    AcceptanceResult
        A dataclass containing the results.
    """
    new_energy = _safe_float(best_docking_score)
    old_energy = _safe_float(old_docking_score)
    delta_e = new_energy - old_energy

    # Guard against zero/negative T
    T_eff = max(float(T), 1e-12)
    p_acc = min(1.0, float(np.exp(-delta_e / T_eff)))
    x = float(rng.uniform(*xrange))

    accepted = (delta_e < 0.0) or (x <= p_acc)

    # If accepted, update ref_seq and "old" scores to new
    if accepted:
        new_ref_seq = mutate_seq
        new_old_score = new_energy
        new_old_pos = pos
        new_old_i = i
        new_old_pre = best_docking_score_pre
        new_old_post = best_docking_score
        logger.info(
            "Mutation ACCEPTED (iter=%d pos=%s) ΔE=%.4f T=%.4f Pacc=%.4f x=%.4f",
            i, pos, delta_e, T_eff, p_acc, x
        )
    else:
        new_ref_seq = ref_seq
        new_old_score = old_energy
        new_old_pos = None
        new_old_i = None
        new_old_pre = old_docking_score_pre
        new_old_post = old_docking_score
        logger.info(
            "Mutation REJECTED (iter=%d pos=%s) ΔE=%.4f T=%.4f Pacc=%.4f x=%.4f",
            i, pos, delta_e, T_eff, p_acc, x
        )

    # Append CSV row (keep your original schema expectations)
    # Note: init_scores_file() should ensure header; we still append robustly.
    with open(DOCKING_SCORES_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{i},{pos},{new_ref_seq},{_safe_float(new_old_pre)},{_safe_float(new_old_post)},{delta_e}\n"
        )

    return AcceptanceResult(
        ref_seq=new_ref_seq,
        old_docking_score=_safe_float(new_old_score),
        old_pos=new_old_pos,
        old_i=new_old_i,
        accepted=accepted,
        delta_e=float(delta_e),
        p_acc=float(p_acc),
        x=float(x),
    )


def plot_docking_scores(docking_scores_file: os.PathLike | str, logger, output_file: Optional[os.PathLike | str] = None) -> None:
    """
        Plot score_pre_simulation and score_post_simulation vs iteration.
        
        Parameters
        ----------
        docking_scores_file : os.PathLike or str
            Path to the CSV file containing docking scores and mutation information.
        logger : logging.Logger
            Logger for logging information and warnings.
        output_file : Optional[os.PathLike or str]
            Optional path to save the plot image. If None, saves as "docking_scores.png" in the same directory as the CSV file.
    """
    docking_scores_file = str(docking_scores_file)
    if not os.path.exists(docking_scores_file):
        logger.warning("[plot_docking_scores] File not found: %s", docking_scores_file)
        return

    try:
        df = pd.read_csv(docking_scores_file)
    except Exception:
        logger.exception("[plot_docking_scores] Failed to read CSV: %s", docking_scores_file)
        return

    required_cols = {"iteration", "score_pre_simulation", "score_post_simulation"}
    if not required_cols.issubset(df.columns):
        logger.warning(
            "[plot_docking_scores] Missing required columns. Found: %s",
            list(df.columns),
        )
        return

    plt.figure(figsize=(8, 5))
    plt.plot(
        df["iteration"],
        df["score_pre_simulation"],
        marker="o",
        linestyle="-",
        color="g",
        label="Score Pre-Simulation",
    )
    plt.plot(
        df["iteration"],
        df["score_post_simulation"],
        marker="s",
        linestyle="-",
        color="b",
        label="Score Post-Simulation",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Score (kcal/mol)")
    plt.title("Pre-Simulation vs Post-Simulation Docking Scores")
    plt.legend()
    plt.grid(True)

    if output_file is None:
        output_file = os.path.join(os.path.dirname(docking_scores_file), "docking_scores.png")

    try:
        plt.savefig(str(output_file), dpi=300, bbox_inches="tight")
        logger.info("[plot_docking_scores] Plot saved: %s", output_file)
    except Exception:
        logger.exception("[plot_docking_scores] Failed saving plot to: %s", output_file)
    finally:
        plt.close()


# -------------------
# Main loop
# -------------------

def run_loop(
    args: Any,
    paths: Dict[str, Any],
    logger,
    receptor_path: os.PathLike | str,
    ref_ligand_path: os.PathLike | str,
    DOCKING_SCORES_FILE: os.PathLike | str,
    changing: bool = False,
) -> None:
    """
    Main Monte Carlo loop.

    Parameters
    ----------
    args : Any
        Parsed command-line arguments, expected to have attributes like steps, out_base, etc.
    paths : Dict[str, Any]
        Dictionary containing various paths used in the pipeline (e.g., CHECKPOINT_DIR, JSON_DIR, etc.).
    logger : logging.Logger
        Logger for logging information and errors.
    receptor_path : os.PathLike or str
        Path to the receptor PDB file.
    ref_ligand_path : os.PathLike or str
        Path to the reference ligand PDB file.
    DOCKING_SCORES_FILE : os.PathLike or str
        Path to the CSV file where docking scores and mutation information will be logged.
    changing : bool
        If True and n_mut > 1, gradually decreases the number of mutations every 50 iterations.
    """
    seed = _get_seed_from_args(args)
    rng = default_rng(seed)
    if seed is not None:
        logger.info("RNG seeded with seed=%s", seed)
    else:
        logger.info("RNG seed not provided; using nondeterministic seed")

    xrange = (0.0, 1.0)

    T_0 = float(args.temperature)
    T_min = 0.4
    l_exp_slow = 0.01

    # Load latest checkpoint (external function)
    ref_seq, old_pos, old_i, old_docking_score, start_iter, T = load_latest_checkpoint(
        paths["CHECKPOINT_DIR"],
        args.ref_seq,
        "ref",
        "seq",
        -7.97,
        T_0,
        logger=logger
    )
    old_docking_score = _safe_float(old_docking_score)
    old_docking_score_pre = old_docking_score  # keep existing behavior

    logger.info(
        "Starting loop from iteration=%d | ref_seq=%s | old_score=%.3f",
        start_iter, ref_seq, old_docking_score
    )

    for i in range(int(start_iter), int(args.iter)):
        iter_start = time.time()

        # Temperature schedule
        T = max(T_0 * float(np.exp(-l_exp_slow * i)), T_min)

        # Mutation schedule
        if changing and int(args.n_mut) > 1:
            n_mut_current = max(1, int(args.n_mut) - (i // 50))
        else:
            n_mut_current = int(args.n_mut)

        logger.info("--- Iter %d | T=%.3f | n_mut=%d ---", i, T, n_mut_current)

        mutate_seq, pos = propose_mutation(ref_seq, args.keep_pos, AMINO_ACIDS, n_mut_current)
        logger.info("Proposed mutation pos=%s | seq=%s", pos, mutate_seq)

        # --- Pre-simulation docking ---
        try:
            best_pre = run_docking(
                i=i,
                pos=pos,
                mutate_seq=mutate_seq,
                JSON_DIR=paths["JSON_DIR"],
                LIG_INPUT_FOLDER=paths["LIG_INPUT_FOLDER"],
                LIG_OUTPUT_FOLDER=paths["LIG_OUTPUT_FOLDER"],
                MINIMIZED_OUTPUT_PATH=paths["MINIMIZED_OUTPUT_PATH"],
                DOCKING_OUT_PATH=paths["DOCKING_OUT_PATH"],
                receptor_path=receptor_path,
                ref_ligand_path=ref_ligand_path,
                base_dir=args.base_dir,
                logger=logger,
            )
        except Exception:
            logger.exception("Docking error (iter=%d pos=%s). Saving checkpoint and continuing.", i, pos)
            save_checkpoint(i, paths["CHECKPOINT_DIR"], ref_seq, old_pos, old_i, old_docking_score, T, interval=10, logger=logger)
            continue

        # --- Only proceed if pre-score passes threshold ---
        if best_pre < -5.0:
            try:
                best_post = run_full_simulation(
                    i=i,
                    pos=pos,
                    receptor_path=receptor_path,
                    OUTPUT_PDB=paths["OUTPUT_PDB"],
                    DOCKING_IN_DIR=paths["DOCKING_IN_DIR"],
                    MINIMIZED_OUTPUT_PATH=paths["MINIMIZED_OUTPUT_PATH"],
                    DOCKING_OUT_PATH=paths["DOCKING_OUT_PATH"],
                    args=args,
                    logger=logger,
                    base_dir=args.base_dir,
                )

                acc = accept_or_reject(
                    i=i,
                    pos=pos,
                    mutate_seq=mutate_seq,
                    best_docking_score_pre=best_pre,
                    best_docking_score=best_post,
                    old_docking_score_pre=old_docking_score_pre,
                    old_docking_score=old_docking_score,
                    ref_seq=ref_seq,
                    T=T,
                    rng=rng,
                    xrange=xrange,
                    DOCKING_SCORES_FILE=DOCKING_SCORES_FILE,
                    logger=logger,
                )

                ref_seq = acc.ref_seq
                old_docking_score = acc.old_docking_score
                old_pos = acc.old_pos
                old_i = acc.old_i

                # If accepted, update old_pre to the accepted pre score; else keep previous
                if acc.accepted:
                    old_docking_score_pre = best_pre

            except Exception:
                logger.exception("Full simulation error (iter=%d pos=%s). Saving checkpoint and continuing.", i, pos)
                save_checkpoint(i, paths["CHECKPOINT_DIR"], ref_seq, old_pos, old_i, old_docking_score, T, interval=10, logger=logger)
                continue
        else:
            logger.info(
                "Pre-score threshold not met (iter=%d pos=%s): best_pre=%.3f >= -5.0. Skipping MD.",
                i, pos, best_pre
            )

        # Plot + checkpoint every iteration (keeps original behavior)
        plot_docking_scores(DOCKING_SCORES_FILE, logger=logger)
        save_checkpoint(i, paths["CHECKPOINT_DIR"], ref_seq, old_pos, old_i, old_docking_score, T, interval=10, logger=logger)
        log_time(iter_start, time.time(), i, logger)


# -------------------
# MAIN
# -------------------

def main() -> int:
    args = parse_args()
    paths, base = setup_paths(args)

    DOCKING_SCORES_FILE = Path(paths["DOCKING_SCORES"]) / "docking_scores.csv"
    init_scores_file(DOCKING_SCORES_FILE, args.ref_seq, logger=logger)

    OUTPUT_LOG = Path(base) / "log.log"
    logger = setup_logger(OUTPUT_LOG)

    receptor_path, ref_ligand_path = setup_refs(args, paths, base, logger=logger)

    logger.info("Starting pipeline")
    logger.info("Base dir: %s", args.base_dir)
    logger.info("Scores CSV: %s", DOCKING_SCORES_FILE)
    logger.info("Receptor: %s", receptor_path)
    logger.info("Reference ligand: %s", ref_ligand_path)

    run_loop(args, paths, logger, receptor_path, ref_ligand_path, DOCKING_SCORES_FILE)

    # run_full_postprocess(base_dir=".", global_results=".")

    logger.info("Pipeline completed successfully.")
    print("\nPipeline completed successfully.")
    print(f"Global results: {paths['GLOBAL_RESULTS']}")
    print(f"Top10 contacts: {paths['TOP_10']}")
    print(f"Log file:       {OUTPUT_LOG}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())