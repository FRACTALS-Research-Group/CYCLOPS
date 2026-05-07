#
# Copyright (c) 2026 FRACTALS Research Group
# Visit the Research Group website for more information: https://fractals.group/
#

"""
    PDB processing utilities.
"""

import os
import numpy as np

def split_pdb_by_chain(input_pdb: str, output_pdb_ab: str, output_pdb_c: str, logger):
    """
        Split a PDB file into two separate files based on chain IDs:
        - One file containing chains A and B
        - One file containing chain C
        
        Parameters
        ----------
        input_pdb : str
            Path to the input PDB file.
        output_pdb_ab : str
            Path to the output PDB file containing chains A and B.
        output_pdb_c : str
            Path to the output PDB file containing chain C.
        logger : logging.Logger
            Logger for logging information and errors.
    """
    # TODO: check chains A, B, C 
    os.makedirs(os.path.dirname(output_pdb_ab) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_pdb_c) or ".", exist_ok=True)

    with open(input_pdb, "r", encoding="utf-8") as infile, \
        open(output_pdb_ab, "w", encoding="utf-8") as out_ab, \
        open(output_pdb_c, "w", encoding="utf-8") as out_c:

        for line in infile:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21]  # chain ID column
                if chain_id == "A":
                    out_ab.write(line)
                elif chain_id == "B":
                    out_c.write(line)
            else:
                out_ab.write(line)
                out_c.write(line)

    logger.info("Split PDB by chain: %s -> (%s, %s)", input_pdb, output_pdb_ab, output_pdb_c)

def parse_pdb(file_path: str) -> np.ndarray:
    """
        Parse ATOM/HETATM coordinates from a PDB and return bounding-box center.
        
        Parameters
        ----------
        file_path : str
            Path to the PDB file to parse.

        Returns
        -------
        center : np.ndarray shape (3,)
        The geometric center of the bounding box defined by the ATOM/HETATM coordinates.
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

    coords = np.asarray(coordinates, dtype=float)
    min_coord = np.min(coords, axis=0)
    max_coord = np.max(coords, axis=0)
    center = (min_coord + max_coord) / 2.0
    return center

def prepare_ligand(ref_path: str, new_path: str, logger):
    """
        Translate ligand coordinates so that ligand center matches the reference box center.

        Parameters
        ----------
        ref_path : str
            Reference PDB (defines box center).
        new_path : str
            Ligand PDB to be translated in-place.
        logger : logging.Logger
            Logger for logging information and errors.
    """
    box_center = parse_pdb(ref_path)
    lig_center = parse_pdb(new_path)

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

                # Preserve PDB fixed-width formatting around coordinates
                new_line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                file.write(new_line)
            else:
                file.write(line)

    logger.info("Ligand translated in-place: %s", new_path)

