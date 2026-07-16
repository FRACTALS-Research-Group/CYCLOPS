# import os, shutil
# import numpy as np
# import pandas as pd
# from collections import defaultdict
# import MDAnalysis as mda
# from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
# import time
# from functions.setup import setup_logger

# outbase = "end"

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from typing import Any, Optional, Set

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis

from functions.clustering import get_most_probable_conformation
from functions.pdb_converters import pdb2pdbqt_rece, pdb2pdbqt_lig
from functions.pdb_process import split_pdb_by_chain
from functions.setup import setup_logger
from functions.simulate_complex import simulate_complex
from functions.vinadock import vina_dock


OUTBASE = "end"

def ensure_dir(path: str, logger: Any) -> None:
    """
        Ensure that a directory exists, creating it if necessary.
        
        Parameters
        ----------
        path : str
            The directory path to ensure exists.
        logger : Any
            Logger object for logging the directory creation.
    """
    os.makedirs(path, exist_ok=True)
    logger.info("Ensured directory exists: %s", path)

def log_time(start: float, end: float, logger: Any) -> None:
    """
        Log the elapsed time between start and end timestamps in HH:MM:SS format.

        Parameters
        ----------
        start : float
            The start timestamp.
        end : float
            The end timestamp.
        logger : Any
            Logger object for logging the operation.
    """
    elapsed = end - start
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    logger.info("Completed in %02d:%02d:%02d", int(hrs), int(mins), int(secs))

def run_analysis(base: str, outbase: str, logger: Any) -> None:
    """
        Run the analysis script to process the simulation trajectory and generate reimaged outputs. 
        
        Parameters
        ----------
        base : str
            The base directory where the input files are located and output will be saved.
        outbase : str
            The base name for the input and output files.
        logger : Any
            Logger object for logging the analysis process.
        
        Raises
        ------        
        RuntimeError
            If the analysis script fails with a non-zero return code.
    """
    command = [
        "python",
        "./functions/analyse.py",
        "-p",
        f"{base}/{outbase}_system_minimised.pdb",
        "-t",
        f"{base}/{outbase}_system_traj.dcd",
        "-o",
        f"{base}/{outbase}_system_reimaged",
        "-r",
    ]
    logger.info("Running analysis: %s", " ".join(command))
    ret = subprocess.call(command)
    if ret != 0:
        raise RuntimeError(f"analyse.py failed with return code {ret}")

def reimage_trajectory(base: str, outbase: str, logger: Any) -> None:
    """
        Reimage the trajectory using mdconvert to generate a reimaged PDB trajectory and reference structure.
        
        Parameters
        ----------
        base : str
            The base directory where the input files are located and output will be saved.
        outbase : str
            The base name for the input and output files.
        logger : Any
            Logger object for logging the reimaging process.
       
        Raises
        ------
        RuntimeError
            If the mdconvert command fails with a non-zero return code.
    """

    command = [
        "mdconvert",
        f"{base}/{outbase}_system_reimaged.dcd",
        "-o",
        f"{base}/{outbase}_system_traj_reimaged.pdb",
        "-t",
        f"{base}/{outbase}_system_reimaged.pdb",
    ]
    logger.info("Reimaging trajectory: %s", " ".join(command))
    ret = subprocess.call(command)
    if ret != 0:
        raise RuntimeError(f"mdconvert failed with return code {ret}")

def parse_pdb_center(file_path: str) -> np.ndarray:
    """
        Parse a PDB file to calculate the geometric center of the atomic coordinates.
        
        Parameters
        ----------
        file_path : str
            Path to the PDB file to parse.
        Returns
        -------
        np.ndarray
            A 3-element array containing the x, y, z coordinates of the geometric center.
        
        Raises
        ------
        ValueError
            If no atomic coordinates are found in the PDB file.
    """
    coords = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                coords.append([float(parts[6]), float(parts[7]), float(parts[8])])
    if not coords:
        raise ValueError(f"No coordinates found in {file_path}")
    coords_arr = np.asarray(coords, dtype=float)
    return (coords_arr.min(axis=0) + coords_arr.max(axis=0)) / 2.0


def change_pdb(ref_path: str, new_path: str, logger: Any) -> None:
    """
        Translate coordinates in new_path so that its center matches ref_path center.
        
        Parameters
        ----------
        ref_path : str
            Path to the reference PDB file whose center will be used as the target.
        new_path : str
            Path to the PDB file to be translated.
        logger : Any
            Logger object for logging the translation process.
    """
    box_center = parse_pdb_center(ref_path)
    lig_center = parse_pdb_center(new_path)
    v_trans = lig_center - box_center
    dx, dy, dz = v_trans
    logger.info("Box center: %s | Ligand center: %s | Translation: %s", box_center, lig_center, v_trans)

    with open(new_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(new_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                x = float(parts[6]) - dx
                y = float(parts[7]) - dy
                z = float(parts[8]) - dz
                f.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
            else:
                f.write(line)  


def contacts_single_system(
    pdb_path: str,
    traj_path: str,
    output_dir: str,
    logger: Any,
    cutoff_contact: float = 4.5,
    cutoff_hydrophobic: float = 5.0,
    hydrophobic_resnames: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Analyse contacts between peptide (chainID B) and receptor (chainID A) for one system.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file containing the reference structure.
    traj_path : str
        Path to the trajectory file (DCD) containing the simulation frames.
    output_dir : str
        Directory where the output CSV and copied files will be saved.
    logger : Any
        Logger object for logging the analysis process.
    cutoff_contact : float, optional
        Distance cutoff (in Å) for defining atomic contacts (default is 4.5 Å).
    cutoff_hydrophobic : float, optional
        Distance cutoff (in Å) for defining hydrophobic interactions (default is 5.0 Å).
    hydrophobic_resnames : Optional[Set[str]], optional
        Set of residue names considered hydrophobic (default includes common hydrophobic residues).
    
    Returns
    -------
    pd.DataFrame
        A DataFrame summarizing the contact analysis for each residue in the peptide, including counts of
    """

    if hydrophobic_resnames is None:
        hydrophobic_resnames = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Contact analysis for: pdb=%s traj=%s", pdb_path, traj_path)

    u = mda.Universe(pdb_path, traj_path)
    peptide = u.select_atoms("chainID B")
    receptor = u.select_atoms("chainID A")

    # --- 1. Atomic contacts ---
    contact_frames = defaultdict(set)
    for ts in u.trajectory:
        for resC in peptide.residues:
            for atomC in resC.atoms:
                dists = np.linalg.norm(receptor.positions - atomC.position, axis=1)
                if np.any(dists < cutoff_contact):
                    contact_frames[resC.resid].add(ts.frame)
                    break
    contact_counts = {k: len(v) for k, v in contact_frames.items()}

    # --- 2. H-bonds ---
    has_hydrogens = len(u.select_atoms("name H*")) > 0
    hydrogens_sel = "name H* and chainID C" if has_hydrogens else None

    h = HydrogenBondAnalysis(
        universe=u,
        donors_sel="protein and chainID B",
        hydrogens_sel=hydrogens_sel,
        acceptors_sel="protein and (chainID A)",
        d_a_cutoff=3.5,
        d_h_a_angle_cutoff=150,
    )
    h.run()
    hbonds = h.results.hbonds
    hb_df = pd.DataFrame(
        hbonds,
        columns=["frame", "donor_index", "hydrogen_index", "acceptor_index", "distance", "angle"],
    )
    if not hb_df.empty:
        donor_atoms = u.atoms[hb_df["donor_index"].astype(int).values]
        hb_df["donor_resid"] = [atom.resid for atom in donor_atoms]

    hbond_frames = defaultdict(set)
    for _, row_hb in hb_df.iterrows():
        hbond_frames[row_hb["donor_resid"]].add(row_hb["frame"])
    hbond_counts = {k: len(v) for k, v in hbond_frames.items()}

    # --- 3. Hydrophobic interactions ---
    hydrophobic_frames = defaultdict(set)
    for ts in u.trajectory:
        for resC in peptide.residues:
            if resC.resname not in hydrophobic_resnames:
                continue
            for atomC in resC.atoms:
                dists = np.linalg.norm(receptor.positions - atomC.position, axis=1)
                if np.any(dists < cutoff_hydrophobic):
                    hydrophobic_frames[resC.resid].add(ts.frame)
                    break
    hydrophobic_counts = {k: len(v) for k, v in hydrophobic_frames.items()}

    # --- 4. Summary table ---
    all_resids = set(contact_counts) | set(hbond_counts) | set(hydrophobic_counts)
    data = []
    for resid in sorted(all_resids):
        res_atoms = peptide.select_atoms(f"resid {resid}")
        resname = res_atoms[0].resname if len(res_atoms) > 0 else "UNK"
        contacts = contact_counts.get(resid, 0)
        hb = hbond_counts.get(resid, 0)
        hydroph = hydrophobic_counts.get(resid, 0)
        total = contacts + hb + hydroph

        if hb >= max(contacts, hydroph):
            main = "H-bond"
        elif hydroph >= max(contacts, hb):
            main = "hydrophobic"
        else:
            main = "contact"

        data.append([resid, resname, contacts, hb, hydroph, total, main])

    df = pd.DataFrame(
        data,
        columns=[
            "Residue ID",
            "Resname",
            "Atomic Contacts",
            "H-bonds",
            "Hydrophobic",
            "Total Interactions",
            "Dominant Type",
        ],
    ).sort_values(by="Residue ID")

    # --- 5. Save outputs ---
    csv_out = os.path.join(output_dir, "contacts.csv")
    df.to_csv(csv_out, index=False)

    shutil.copy2(pdb_path, os.path.join(output_dir, os.path.basename(pdb_path)))
    shutil.copy2(traj_path, os.path.join(output_dir, os.path.basename(traj_path)))

    logger.info("Contact analysis complete. CSV saved: %s", csv_out)
    return df         
    
def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the script.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser for the script.
    """
    parser = argparse.ArgumentParser(description="Step 0: prepare and simulate the reference system.")

    parser.add_argument("--ligand_pdb", required=True, help="Path to reference ligand PDB file")
    parser.add_argument("--ref_lig_file", required=True, help="Path to cyclic peptide PDB file")
    parser.add_argument("--input_rece_file", required=True, help="Path to receptor PDB file")
    parser.add_argument("--src_route", required=True, help="Path to the functions directory")
    parser.add_argument("--steps", type=int, default=250000, help="Number of simulation steps")
    parser.add_argument("--base_dir", default="inputs/ref/pdb", help="Base directory for outputs")
    parser.add_argument("--box_size", nargs=3, type=float, default=[20, 20, 20], help="Docking box size (Å)")
    parser.add_argument("--log_file", default="step_0.log", help="Log file path")
    parser.add_argument(
        "--results_path",
        default="step0_results.json",
        help="Path where step 0 docking results will be saved (JSON). "
             "Pass the same path to main.py via --step0_results.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    logger = setup_logger(args.log_file)

    start = time.time()
    ensure_dir(args.base_dir, logger=logger)

    output_base = f"{args.base_dir}/{OUTBASE}"
    logger.info("Output base: %s", output_base)

    # Step 1 — center ligand
    change_pdb(args.ligand_pdb, args.ref_lig_file, logger=logger)

    # Step 2 — simulate ref complex
    simulate_complex(
        protein_pdb_path=args.input_rece_file,
        ligand_pdb_path=args.ref_lig_file,
        base_out=".",
        output_base=output_base,
        steps=args.steps,
        solvate=True,
        annealing=False,
        restrain_protein=False,
        logger=logger,
    )

    # Step 3 — analysis + reimaging
    run_analysis(args.base_dir, OUTBASE, logger=logger)
    reimage_trajectory(args.base_dir, OUTBASE, logger=logger)

    # Step 4 — most probable conformation
    rep_frame, _, _ = get_most_probable_conformation(
        f"{output_base}_system_traj_reimaged.pdb",
        pos="ref",
        i=0,
        output_path=args.base_dir,
        subsample_rate=20,
        logger=logger,
    )
    logger.info("Representative frame: %s", rep_frame)

    # Step 5 — split receptor/ligand
    rece_pdb = f"{args.base_dir}/rep_rece_{OUTBASE}_0.pdb"
    lig_pdb = f"{args.base_dir}/rep_lig_{OUTBASE}_0.pdb"
    split_pdb_by_chain(f"{args.base_dir}/rep_mut_{OUTBASE}_0.pdb", rece_pdb, lig_pdb, logger=logger)

    # Step 6 — convert to pdbqt
    pdb2pdbqt_rece(args.base_dir, args.base_dir, ".", conversion=os.path.basename(rece_pdb), logger=logger)
    pdb2pdbqt_lig(args.base_dir, args.base_dir, ".", conversion=os.path.basename(lig_pdb), logger=logger)

    # Step 7 — docking ref system
    best_score, _poses_mean = vina_dock(
        f"{args.base_dir}/rep_rece_{OUTBASE}_0.pdbqt",
        f"{args.base_dir}/rep_lig_{OUTBASE}_0.pdbqt",
        f"{args.base_dir}/rep_lig_{OUTBASE}_0.pdbqt",
        f"{args.base_dir}/min_{OUTBASE}_0.pdbqt",
        f"{args.base_dir}/out_{OUTBASE}_0.pdbqt",
        box_size=args.box_size,
        exhaustiveness=32,
        n_poses=10,
        logger=logger,
    )
    logger.info("Docking score (ref system): %.2f kcal/mol", best_score)

    # Save step-0 results so that main.py can read them instead of using hardcoded values.
    step0_results = {
        "score_pre_simulation": float(best_score),
        "score_post_simulation": float(best_score),
    }
    results_path = os.path.abspath(args.results_path)
    with open(results_path, "w", encoding="utf-8") as _f:
        json.dump(step0_results, _f, indent=2)
    logger.info("Step-0 results saved to: %s", results_path)

    contacts_single_system(
        pdb_path=f"{args.base_dir}/{OUTBASE}_system_reimaged.pdb",
        traj_path=f"{args.base_dir}/{OUTBASE}_system_traj_reimaged.pdb",
        output_dir=f"{args.base_dir}/analysis_contacts",
        logger=logger,
    )

    log_time(start, time.time(), logger=logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
