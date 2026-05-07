from typing import Any, List, Sequence, Tuple

import numpy as np
from vina import Vina

def parse_pdbqt(file_path: str) -> np.ndarray:
    """
        Given the coordinates in a PDBQT file, return the geometric center (bbox center).
        Define the center of the binding pocket (file = ref_ligand) and the center of new ligands.
        Translate the ligand (see change_pdbqt()).

        Parameters
        ----------
        file_path : str
            Path to the PDBQT file to parse.
        Returns
        -------
        np.ndarray
            Geometric center of the coordinates in the PDBQT file as a NumPy array of shape (3,).
        
        Raises
        ------
        ValueError
            If no ATOM/HETATM coordinates are found in the file.
    """
    coordinates = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                x = float(parts[6])
                y = float(parts[7])
                z = float(parts[8])
                coordinates.append((x, y, z))

    if not coordinates:
        raise ValueError(f"No ATOM/HETATM coordinates found in: {file_path}")

    coordinates = np.asarray(coordinates, dtype=float)
    min_coord = np.min(coordinates, axis=0)
    max_coord = np.max(coordinates, axis=0)
    center = (min_coord + max_coord) / 2.0
    return center

def change_pdbqt(ref_path: str, new_path: str, logger: Any) -> np.ndarray:
    """
        Translate ligand coordinates in new_path so that its center matches ref_path center.
        Operates in-place and returns the box center.
        
        Parameters
        ----------
        ref_path : str
            Path to the reference PDBQT file (e.g., reference ligand).
        new_path : str
            Path to the new PDBQT file (e.g., new ligand) to be translated.
        logger : Any
            Logger for logging information and errors.
        
        Returns
        -------
        np.ndarray
            Geometric center of the reference PDBQT file (box center) as a NumPy array of shape (3,).
        
        Raises
        ------
        ValueError
            If no ATOM/HETATM coordinates are found in either file.
    """
    box_center = parse_pdbqt(ref_path)
    lig_center = parse_pdbqt(new_path)

    v_trans = lig_center - box_center
    dx, dy, dz = v_trans

    logger.info("Center of the box set at: %s", box_center)
    logger.info("Center of the ligand set at: %s", lig_center)
    logger.info("Translation vector: %s", v_trans)

    with open(new_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open(new_path, "w", encoding="utf-8") as file:
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                x = float(parts[6]) - dx
                y = float(parts[7]) - dy
                z = float(parts[8]) - dz
                new_line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                file.write(new_line)
            else:
                file.write(line)

    return box_center


def parse_affinity(file_path: str) -> Tuple[float, float]:
    """
        Parse affinities from a Vina output PDBQT file.
        Returns (best, mean).
        
        Parameters
        ----------
        file_path : str
            Path to the Vina output PDBQT file to parse.
        
        Returns
        -------
        Tuple[float, float]
            A tuple containing the best affinity (lowest energy) and the mean affinity across all poses.
        
        Raises
        ------
        ValueError
            If no Vina affinities are found in the file.
    """
    affinities: List[float] = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("REMARK VINA RESULT:"):
                affinity = float(line[22:29])
                affinities.append(affinity)

    if not affinities:
        raise ValueError(f"No Vina affinities found in: {file_path}")

    mean = float(sum(affinities) / len(affinities))
    best = float(min(affinities))
    return best, mean


def vina_dock(
    receptor_path: str,
    ref_ligand_path: str,
    new_ligand_path: str,
    minimized_output_path: str,
    docking_vina_out_path: str,
    logger: Any,
    dock: bool = True,
    box_size: Sequence[float] = (20, 20, 20),
    exhaustiveness: int = 32,
    n_poses: int = 10,
) -> Tuple[float, float]:
    """
        Run Vina scoring/minimization and optionally docking, returning the best and mean docking scores.

        Parameters
        ----------
        receptor_path : str
            Path to the receptor PDBQT file.
        ref_ligand_path : str
            Path to the reference ligand PDBQT file (used for defining the box center).
        new_ligand_path : str
            Path to the new ligand PDBQT file (to be translated and docked).
        minimized_output_path : str
            Path to save the minimized ligand pose PDBQT file.
        docking_vina_out_path : str
            Path to save the docking output PDBQT file containing multiple poses.
        logger : Any
            Logger for logging information and errors.
        dock : bool, optional
            Whether to perform docking (default is True). If False, only scoring and minimization are performed.
        box_size : Sequence[float], optional
            Size of the docking box in Angstroms (default is (20, 20, 20)).
        exhaustiveness : int, optional
            Exhaustiveness parameter for Vina docking (default is 32).
        n_poses : int, optional
            Number of docking poses to generate (default is 10).
        
        Returns
        -------
        Tuple[float, float]
            A tuple containing the best docking score (lowest energy) and the mean docking score across all poses. If docking is not performed, both values will be the score after minimization.
        
        Raises
        ------
        Exception
            If any error occurs during the Vina docking process, including file I/O errors or issues with the Vina library.
    """
    v = Vina(sf_name="vina")

    v.set_receptor(receptor_path)
    logger.info("Receptor set: %s", receptor_path)

    box_center = change_pdbqt(ref_ligand_path, new_ligand_path, logger=logger)

    v.set_ligand_from_file(new_ligand_path)
    v.compute_vina_maps(center=box_center, box_size=list(box_size))

    try:
        energy = float(v.score()[0])
        logger.info("Score before minimization: %.3f (kcal/mol)", energy)
    except Exception:
        logger.warning("Initial scoring failed; retrying with larger box (30,30,30)")
        v.compute_vina_maps(center=box_center, box_size=[30, 30, 30])
        energy = float(v.score()[0])
        logger.info("Score before minimization: %.3f (kcal/mol)", energy)

    energy_minimized = float(v.optimize()[0])
    logger.info("Score after minimization: %.3f (kcal/mol)", energy_minimized)
    v.write_pose(minimized_output_path, overwrite=True)

    if dock:
        v.dock(exhaustiveness=int(exhaustiveness), n_poses=int(n_poses))
        v.write_poses(docking_vina_out_path, n_poses=int(n_poses), overwrite=True)
        poses_best_energy, poses_mean_energy = parse_affinity(docking_vina_out_path)
    else:
        poses_best_energy = energy_minimized
        poses_mean_energy = energy_minimized

    return float(poses_best_energy), float(poses_mean_energy)


def calculate_and_format_docking_scores(mean_docking_score: float, best_docking_score: float) -> Tuple[str, str]:
    """
        Calculate and format docking scores with absolute and relative errors.

        Parameters
        ----------
        mean_docking_score : float
            The mean docking score across all poses.
        best_docking_score : float
            The best docking score (lowest energy).

        Returns
        -------
        Tuple[str, str]
            A tuple containing the formatted docking scores with absolute and relative errors.
    """
    absolute_error = abs(mean_docking_score - best_docking_score)
    relative_error = (absolute_error / mean_docking_score) * 100 if mean_docking_score != 0 else float("inf")

    docking_score_with_absolute_error = f"{best_docking_score:.3f} ± {absolute_error:.3f} kcal/mol"
    docking_score_with_relative_error = f"{best_docking_score:.3f} ± {relative_error:.2f}%"

    return docking_score_with_absolute_error, docking_score_with_relative_error
