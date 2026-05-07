#
# Copyright (c) 2026 FRACTALS Research Group
# Visit the Research Group website for more information: https://fractals.group/
#

"""
    PDB conversion utilities.
"""
from Bio import PDB
import os
from glob import glob
import subprocess


def cif2pdb(input_cif: str, output_pdb: str, logger):
    """
        Convert an mmCIF file to PDB format.
        
        Parameters
        ----------
        input_cif : str
            Path to the input mmCIF file.
        output_pdb : str
            Path where the output PDB file will be saved.
        logger : logging.Logger
            Logger for logging information and errors.
    """
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure("AF3_Model", input_cif)

    io = PDB.PDBIO()
    io.set_structure(structure)

    io.save(output_pdb)
    logger.info("Converted CIF -> PDB: %s -> %s", input_cif, output_pdb)

def pdb2pdbqt_rece(input_folder: str, output_folder: str, base: str, logger, conversion: str = "*.pdb"):
    """
        Convert receptor PDB(s) to PDBQT using ADFRsuite prepare_receptor.
        
        Parameters
        ----------
        input_folder : str
            Directory containing the input PDB files (or a single PDB file named <input_folder>.pdb).
        output_folder : str
            Directory where the output PDBQT files will be saved.
        base : str
            Base path to the ADFRsuite installation.
        logger : logging.Logger
            Logger for logging information and errors.
        conversion : str, optional
            Glob pattern to match PDB files in the input folder (default is "*.pdb").
        
        Raises
        ------
        RuntimeError
            If prepare_receptor fails or if no valid PDB files are found.
    """
    os.makedirs(output_folder, exist_ok=True)
    prepare_receptor = (
        f"{base}/functions/ADFRsuite-1.1dev/ADFRsuite_x86_64Linux_1.1dev/docking/bin/prepare_receptor"
    )

    pdb_files = glob(os.path.join(input_folder, conversion))

    # Keep your original behavior: if glob finds nothing, try "<input_folder>.pdb"
    if not pdb_files:
        fallback = f"{input_folder}.pdb"
        pdb_files = [fallback]
        logger.warning("No files found with pattern %s, trying fallback: %s", conversion, fallback)

    for pdb_file in pdb_files:
        if not os.path.exists(pdb_file):
            logger.warning("Receptor PDB not found, skipping: %s", pdb_file)
            continue

        pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
        pdbqt_file = os.path.join(output_folder, f"{pdb_name}.pdbqt")

        command = [prepare_receptor, "-r", pdb_file, "-o", pdbqt_file]
        ret = subprocess.call(command)

        if ret != 0:
            logger.error("prepare_receptor failed (code=%d): %s", ret, " ".join(command))
            raise RuntimeError(f"prepare_receptor failed with code {ret}")
        logger.info("Converted receptor: %s -> %s", pdb_file, pdbqt_file)


def pdb2pdbqt_lig(input_folder: str, output_folder: str, base: str, logger, conversion: str = "*.pdb"):
    """
        Convert ligand PDB to PDBQT using ADFRsuite prepare_ligand.
        
        Parameters
        ----------
        input_folder : str
            Directory containing the input PDB file (or a single PDB file named <input_folder>.pdb).
        output_folder : str
            Directory where the output PDBQT file will be saved.
        base : str
            Base path to the ADFRsuite installation.
        logger : logging.Logger
            Logger for logging information and errors.
        conversion : str, optional
            Glob pattern to match the PDB file in the input folder (default is "*.pdb").
        
        Raises
        ------
        RuntimeError
            If prepare_ligand fails or if no valid PDB file is found.
    """
    os.makedirs(output_folder, exist_ok=True)
    prepare_ligand = (
        f"{base}/functions/ADFRsuite-1.1dev/ADFRsuite_x86_64Linux_1.1dev/docking/bin/prepare_ligand"
    )

    # Keep original behavior: expects "<input_folder>.pdb"
    pdb_file = f"{input_folder}.pdb"
    if not os.path.exists(pdb_file):
        logger.error("Ligand PDB not found: %s", pdb_file)
        raise FileNotFoundError(f"Ligand PDB not found: {pdb_file}")

    pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]
    pdbqt_file = os.path.join(output_folder, f"{pdb_name}.pdbqt")

    command = [prepare_ligand, "-l", pdb_file, "-o", pdbqt_file]
    ret = subprocess.call(command)

    if ret != 0:
        logger.error("prepare_ligand failed (code=%d): %s", ret, " ".join(command))
        raise RuntimeError(f"prepare_ligand failed with code {ret}")
    logger.info("Converted ligand: %s -> %s", pdb_file, pdbqt_file)



def pdbqt2pdb(input_folder: str, output_folder: str, base: str, pos: str, i: int, logger):
    """
        Convert docking output PDBQT -> PDB using obabel.
        
        Parameters
        ----------
        input_folder : str
            Directory containing the input PDBQT file(s) (expects files named out_{pos}_{i}.pdbqt).
        output_folder : str
            Directory where the output PDB file(s) will be saved.
        base : str
            Base path to the ADFRsuite installation (used to locate obabel).
        pos : str
            Position identifier used in the input file naming pattern.
        i : int
            Iteration identifier used in the input file naming pattern.
        logger : logging.Logger
            Logger for logging information and errors.
    """
    os.makedirs(output_folder, exist_ok=True)
    obabel = f"{base}/functions/ADFRsuite-1.1dev/ADFRsuite_x86_64Linux_1.1dev/docking/bin/obabel"

    pdbqt_files = glob(os.path.join(input_folder, f"out_{pos}_{i}.pdbqt"))
    if not pdbqt_files:
        logger.warning("No PDBQT found in %s with name out_%s_%s.pdbqt", input_folder, pos, i)
        return

    for pdbqt_file in pdbqt_files:
        # --- Keep only MODEL 1 and remove ENDROOT section ---
        with open(pdbqt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        in_first_model = False
        in_section_to_remove = False

        for line in lines:
            if line.startswith("MODEL"):
                model_number = int(line.strip()[6:])
                if model_number == 1:
                    in_first_model = True
                else:
                    in_first_model = False
                    continue

            if line.startswith("ENDROOT"):
                in_section_to_remove = True

            if in_first_model and (not in_section_to_remove):
                new_lines.append(line)

            if line.startswith("ENDMDL"):
                in_section_to_remove = False

        # Overwrite the .pdbqt file with the filtered content
        with open(pdbqt_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        
        pdb_file = os.path.join(output_folder, os.path.basename(pdbqt_file).replace(".pdbqt", ".pdb"))
        if not pdb_file.endswith(".pdb"):
            raise ValueError(f"Invalid output: {pdb_file}")

        command = [obabel, "-ipdbqt", pdbqt_file, "-O", pdb_file]
        ret = subprocess.call(command)

        if ret != 0:
            logger.error("obabel failed (code=%d): %s", ret, " ".join(command))
            raise RuntimeError(f"obabel failed with code {ret}")
        logger.info("Converted PDBQT -> PDB: %s -> %s", pdbqt_file, pdb_file)
