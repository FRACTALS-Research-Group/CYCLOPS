#
# Copyright (c) 2026 FRACTALS Reseasrch Group
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

import time
import traceback

from numpy.random import default_rng
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

from functions.setup import setup_logger, setup_paths, setup_refs, init_scores_file, parse_args, log_time, load_step0_results
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
from functions.postprocess import run_full_postprocess

# -------------------
# Workflow functions
# -------------------
def propose_mutation(ref_seq, keep_pos, amino_acids, n_mut):
    valid_indices = [j for j in range(len(ref_seq)) if j not in keep_pos]
    mutate_seq, pos = random_n_mutation(ref_seq, valid_indices, amino_acids, n=n_mut)
    return mutate_seq, "_".join(map(str, pos))


def run_docking(i, pos, mutate_seq, JSON_DIR, LIG_INPUT_FOLDER, LIG_OUTPUT_FOLDER,
                MINIMIZED_OUTPUT_PATH, DOCKING_OUT_PATH, receptor_path, ref_ligand_path, base_dir, logger):
    json_path = generate_json_files(pos, i, mutate_seq, output_dir=JSON_DIR)
    alphafold3(json_path, output_dir=LIG_INPUT_FOLDER)
    cif2pdb(
        f"{LIG_INPUT_FOLDER}/seq_{pos}_{i}/seq_{pos}_{i}_model.cif",
        f"{LIG_INPUT_FOLDER}/seq_{pos}_{i}/seq_{pos}_{i}.pdb"
    )
    pdb2pdbqt_lig(f"{LIG_INPUT_FOLDER}/seq_{pos}_{i}/seq_{pos}_{i}", f"{LIG_OUTPUT_FOLDER}/seq_{pos}_{i}", base_dir,
                   conversion=f"seq_{pos}_{i}.pdb")
    
    best_score, _ = vina_dock(
        receptor_path, ref_ligand_path,
        f"{LIG_OUTPUT_FOLDER}/seq_{pos}_{i}/seq_{pos}_{i}.pdbqt",
        f"{MINIMIZED_OUTPUT_PATH}/min_{pos}_{i}.pdbqt",
        f"{DOCKING_OUT_PATH}/out_{pos}_{i}.pdbqt",
        box_size=[20,20,20], exhaustiveness=32, n_poses=10
    )
    pdbqt2pdb(f"{DOCKING_OUT_PATH}", f"{DOCKING_OUT_PATH}/pdb", base_dir, pos, i)
    logger.info(f"Docking pre-simulation score={best_score:.2f}")
    return best_score


def run_full_simulation(i, pos, receptor_path, OUTPUT_PDB, DOCKING_IN_DIR,
                        MINIMIZED_OUTPUT_PATH, DOCKING_OUT_PATH, args, logger, base_dir):

    seq_dir = f'{OUTPUT_PDB}/seq_{pos}_{i}'
    if not os.path.exists(seq_dir):
        os.makedirs(seq_dir)

    # --- Simulate complex using the safe simulate_complex ---
    try:
        simulation, system, system_generator = simulate_complex(
            protein_pdb_path=receptor_path,
            ligand_pdb_path=f"{DOCKING_OUT_PATH}/pdb/out_{pos}_{i}.pdb",
            base_out=seq_dir,
            output_base=f'{seq_dir}/{args.out_base}_{pos}_{i}',
            steps=args.steps,
            solvate=True,
            annealing=False,
            restrain_protein=False,
            T_restrain=400,
            temp_start=1000,
            temp_end=310
        )
    except Exception as e:
        logger.error(f"MD simulation failed for iteration {i}, pos {pos}: {e}")
        raise

    # --- Post-MD analysis ---
    run_analysis(args.base_dir, args.out_base, pos, i)
    reimage_trajectory(args.base_dir, args.out_base, pos, i)

    dock_dir = f'{DOCKING_IN_DIR}/seq_{pos}_{i}'
    if not os.path.exists(dock_dir):
        os.makedirs(dock_dir)

    # --- Get representative frame & split chains ---
    representative_frame, _, _ = get_most_probable_conformation(
        f'{seq_dir}/{args.out_base}_{pos}_{i}_traj_reimaged.pdb',
        pos, i, output_path=seq_dir, subsample_rate=20
    )

    split_pdb_by_chain(
        f"{seq_dir}/rep_mut_{pos}_{i}.pdb",
        f"{dock_dir}/rep_rece_{pos}_{i}.pdb",
        f"{dock_dir}/rep_lig_{pos}_{i}.pdb"
    )

    pdb2pdbqt_rece(dock_dir, dock_dir, base_dir, conversion=f'rep_rece_{pos}_{i}.pdb')
    pdb2pdbqt_lig(f"{dock_dir}/rep_lig_{pos}_{i}", dock_dir, base_dir, conversion=f'rep_lig_{pos}_{i}.pdb')

    # --- Docking post-MD ---
    best_score, _ = vina_dock(
        f"{dock_dir}/rep_rece_{pos}_{i}.pdbqt",
        f"{dock_dir}/rep_lig_{pos}_{i}.pdbqt",
        f"{dock_dir}/rep_lig_{pos}_{i}.pdbqt",
        f"{MINIMIZED_OUTPUT_PATH}/min_system_{pos}_{i}.pdbqt",
        f"{DOCKING_OUT_PATH}/out_system_{pos}_{i}.pdbqt",
        dock=False,
        box_size=[20,20,20],
        exhaustiveness=32,
        n_poses=10
    )

    logger.info(f"Docking post-simulation score={best_score:.2f}")
    return best_score


# def K_to_kcal(T_K):
#     kB_kcal_per_molK = 0.0019872  # Boltzmann constant in kcal/(mol·K)
#     return T_K * kB_kcal_per_molK


def accept_or_reject(
    i, pos, mutate_seq,
    best_docking_score_pre, best_docking_score,
    old_docking_score_pre, old_docking_score,
    ref_seq, T, rng, xrange,
    DOCKING_SCORES_FILE
):
    
    new_energy = float(best_docking_score)
    old_energy = float(old_docking_score)
    delta_e = new_energy - old_energy
    
    # T_kcal = K_to_kcal(T) 
    Pacc = min(1, np.exp(-delta_e / T))

    x = rng.uniform(*xrange)

    if delta_e < 0 or (delta_e > 0 and x <= Pacc):
        print("Mutation accepted")
        ref_seq = mutate_seq
        old_pos = pos
        old_i = i
        old_docking_score_pre = best_docking_score_pre
        old_docking_score = best_docking_score
        with open(DOCKING_SCORES_FILE, "a") as f:
            f.write(f"{i},{pos},{ref_seq},{best_docking_score_pre},{best_docking_score},{delta_e}\n")
    else:
        print("Mutation rejected")
        old_pos = None
        old_i = None
        with open(DOCKING_SCORES_FILE, "a") as f:
            f.write(f"{i},{pos},{ref_seq},{old_docking_score_pre},{old_docking_score},{delta_e}\n")

    return ref_seq, old_docking_score, old_pos, old_i


def plot_docking_scores(docking_scores_file, output_file=None):
    if not os.path.exists(docking_scores_file):
        print(f"[plot_docking_scores] File non trovato: {docking_scores_file}")
        return

    df = pd.read_csv(docking_scores_file)

    required_cols = {"iteration", "score_pre_simulation", "score_post_simulation"}
    if not required_cols.issubset(df.columns):
        print(f"[plot_docking_scores] Colonne mancanti nel CSV. Trovate: {df.columns}")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df["iteration"], df["score_pre_simulation"],
             marker='o', linestyle='-', color='g', label="Score Pre-Simulation")
    plt.plot(df["iteration"], df["score_post_simulation"],
             marker='s', linestyle='-', color='b', label="Score Post-Simulation")

    plt.xlabel("Iteration")
    plt.ylabel("Score (kcal/mol)")
    plt.title("Pre-Simulation vs Post-Simulation Docking Scores")
    plt.legend()
    plt.grid()

    if output_file is None:
        output_file = os.path.join(os.path.dirname(docking_scores_file), "docking_scores.png")

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[plot_docking_scores] Plot salvato in: {output_file}")


# -------------------
# Main loop
# -------------------
def run_loop(args, paths, logger, receptor_path, ref_ligand_path, DOCKING_SCORES_FILE, changing=False):
    rng = default_rng()
    xrange = (0.0, 1.0)
    T_0 = args.temperature
    T_min = 0.4
    l_exp_slow = 0.01
    amino_acids = list("ARNDQEGHILKMFPSTWYV")

    ref_seq, old_pos, old_i, old_docking_score, start_iter, T = load_latest_checkpoint(
        paths["CHECKPOINT_DIR"], args.ref_seq, "ref", "seq",
        load_step0_results(paths["STEP0_RESULTS"], logger)["score_post_simulation"],
        T_0
    )
    old_docking_score_pre = old_docking_score
    logger.info(f"Starting from iteration {start_iter}, ref_seq={ref_seq}")
    
    for i in range(start_iter, args.iter):
        iter_start = time.time()
        T = max(T_0 * np.exp(-l_exp_slow * i), T_min)

        if changing and args.n_mut > 1:  # non scendere sotto 1
            n_mut_current = max(1, args.n_mut - (i // 50))
        else:
            n_mut_current = args.n_mut

        logger.info(f"\n--- Iter {i} | T={T:.3f} | n_mut={n_mut_current} ---")

        mutate_seq, pos = propose_mutation(ref_seq, args.keep_pos, amino_acids, n_mut_current)
        logger.info(f"Proposed mutation at {pos} → {mutate_seq}")

        try:
            best_pre = run_docking(i, pos, mutate_seq, paths["JSON_DIR"], paths["LIG_INPUT_FOLDER"],
                                   paths["LIG_OUTPUT_FOLDER"], paths["MINIMIZED_OUTPUT_PATH"], paths["DOCKING_OUT_PATH"],
                                   receptor_path, ref_ligand_path, args.base_dir, logger)
        except Exception as e:
            logger.error(f"Docking error: {e}")
            traceback.print_exc()
            save_checkpoint(i, paths["CHECKPOINT_DIR"], ref_seq, old_pos, old_i, old_docking_score, T, interval=10)
            continue

        if best_pre < -5:
            try:
                best_post = run_full_simulation(i, pos, receptor_path, paths["OUTPUT_PDB"], paths["DOCKING_IN_DIR"],
                                                paths["MINIMIZED_OUTPUT_PATH"], paths["DOCKING_OUT_PATH"], args, logger, args.base_dir)
                ref_seq, old_docking_score, old_pos, old_i = accept_or_reject(
                    i, pos, mutate_seq, best_pre, best_post,
                    old_docking_score_pre, old_docking_score,
                    ref_seq, T, rng, xrange, DOCKING_SCORES_FILE
                )
            except Exception as e:
                logger.error(f"Full simulation error: {e}")
                traceback.print_exc()
                save_checkpoint(i, paths["CHECKPOINT_DIR"], ref_seq, old_pos, old_i, old_docking_score, T, interval=10)
                continue
        
        plot_docking_scores(DOCKING_SCORES_FILE)
        save_checkpoint(i, paths["CHECKPOINT_DIR"], ref_seq, old_pos, old_i, old_docking_score, T, interval=10)
        log_time(iter_start, time.time(), i, logger)


# -------------------
# MAIN
# -------------------
if __name__ == "__main__":
    args = parse_args()
    paths, base = setup_paths(args)

    OUTPUT_LOG = base / "log.log"
    logger = setup_logger(OUTPUT_LOG)

    DOCKING_SCORES_FILE = paths["DOCKING_SCORES"] / "docking_scores.csv"
    init_scores_file(DOCKING_SCORES_FILE, args.ref_seq, paths["STEP0_RESULTS"], logger)
    
    receptor_path, ref_ligand_path = setup_refs(args, paths, base)

    run_loop(args, paths, logger, receptor_path, ref_ligand_path, DOCKING_SCORES_FILE) 

    # run_full_postprocess(base_dir=".", global_results=".")

    logger.info("Pipeline completed successfully.")
    print("\nPipeline completed successfully.")
    print(f"Global results: {paths['GLOBAL_RESULTS']}")
    print(f"Top10 contacts: {paths['TOP_10']}")
    print(f"Log file:       {OUTPUT_LOG}\n")