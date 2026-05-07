import argparse
from typing import Any, Optional

import mdtraj as md
import plotly.graph_objects as go

import logging



# Typical usage:
# python functions/analyse.py -p path/to/protein.pdb -t path/to/trajectory.dcd -o output_base_name --remove-waters


def build_parser() -> argparse.ArgumentParser:
    """
        Builds the argument parser for the analyse function.

        Returns
        -------
        argparse.ArgumentParser
            The configured argument parser for the analyse function.
    """
    parser = argparse.ArgumentParser(description="analyse")
    parser.add_argument("-p", "--protein", required=True, help="Protein PDB file")
    parser.add_argument("-t", "--trajectory", required=True, help="Trajectory DCD file")
    parser.add_argument("-o", "--output", required=True, help="Output base name")
    parser.add_argument("-r", "--remove-waters", action="store_true", help="Remove waters, salts etc.")
    return parser

def run_analysis(protein_pdb: str, trajectory_dcd: str, out_base: str, *,
    remove_waters: bool = False, logger: logging.Logger
    ) -> None:
    """
        Analyze a trajectory by re-imaging, calculating RMSD, and plotting results.
        It produces:
        - A re-imaged PDB file (out_base.pdb)
        - A re-imaged DCD trajectory (out_base.dcd)
        - An SVG plot of ligand and backbone RMSD over time (out_base.svg)
        
        Parameters
        ----------
        protein_pdb : str
            Path to the input protein PDB file (used as topology for the trajectory).
        trajectory_dcd : str
            Path to the input trajectory DCD file.
        out_base : str
            Base name for output files (without extension).
        remove_waters : bool, optional
            Whether to remove water, salts, and lipids from the trajectory before analysis (default is False).
        logger : logging.Logger
            Logger for logging information and errors.
    """
    logger.info("Reading trajectory: %s (top=%s)", trajectory_dcd, protein_pdb)
    t = md.load(trajectory_dcd, top=protein_pdb)
    t.image_molecules(inplace=True)

    if remove_waters:
        logger.info("Removing waters/salts/lipids")
        t = t.atom_slice(t.top.select("not resname HOH POPC CL NA"))

    logger.info("Realigning trajectory (superpose to frame 0 using protein atoms)")
    prot = t.top.select("protein")
    t.superpose(t[0], atom_indices=prot)

    logger.info("Writing re-imaged PDB: %s.pdb", out_base)
    t[0].save(out_base + ".pdb")

    logger.info("Writing re-imaged trajectory: %s.dcd", out_base)
    t.save(out_base + ".dcd")

    logger.info("Number of frames: %d", t.n_frames)

    atoms_lig = t.topology.select("chainid 1")
    logger.info("Ligand atoms: %d", len(atoms_lig))
    rmsds_lig = md.rmsd(t, t, frame=0, atom_indices=atoms_lig, parallel=True, precentered=False)

    atoms_bck = t.topology.select("chainid 0 and backbone")
    logger.info("Backbone atoms: %d", len(atoms_bck))
    rmsds_bck = md.rmsd(t, t, frame=0, atom_indices=atoms_bck, parallel=True, precentered=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t.time, y=rmsds_lig, mode="lines", name="Ligand"))
    fig.add_trace(go.Scatter(x=t.time, y=rmsds_bck, mode="lines", name="Backbone"))
    fig.update_layout(title="Trajectory for " + trajectory_dcd, xaxis_title="Frame", yaxis_title="RMSD")

    out_svg = out_base + ".svg"
    logger.info("Writing RMSD output: %s", out_svg)
    fig.write_image(out_svg)

def main(argv: Optional[list[str]] = None) -> int:
    """
        Main function to execute the analysis.

        Parameters
        ----------
        argv : Optional[list[str]]
            List of command-line arguments. If None, uses sys.argv. (default is None)
        Returns
        -------
        int
            Exit code (0 for success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("analyse")

    logger.info("analyse args: %s", args)
    run_analysis(
        protein_pdb=args.protein,
        trajectory_dcd=args.trajectory,
        out_base=args.output,
        remove_waters=args.remove_waters,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())