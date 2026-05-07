from collections import Counter
from typing import Any, Dict, Tuple

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns


def write_topology(input_pdb: str, output_pdb: str, logger) -> str:
    """
        Write the first frame of a PDB/trajectory to a single-frame PDB topology file.
        
        Parameters
        ----------
        input_pdb : str
            Path to the input PDB file.
        output_pdb : str
            Path to the output PDB file where the topology will be saved.
        logger : Any
            Logger for logging information and errors.
        
        Returns
        -------
        str
            Path to the generated topology PDB file.
    """

    u = mda.Universe(input_pdb)
    u.trajectory[0]

    with mda.Writer(output_pdb, multiframe=False) as W:
        W.write(u)

    logger.info("Topology written: %s -> %s", input_pdb, output_pdb)
    return output_pdb


def get_most_probable_conformation(
    trajectory_file: str,
    pos: str,
    i: int,
    output_path: str,
    logger: Any,
    selection: str = "backbone",
    eps: float = 1.0,
    min_samples: int = 5,
    plot: bool = True,
    subsample_rate: int = 1000,
) -> Tuple[int, int, Dict[int, int]]:
    """
        Cluster frames using DBSCAN over an RMSD matrix (subsampled) and write
        the representative structure (first frame of largest cluster).

        Parameters
        ----------
        trajectory_file : str
            Path to the input trajectory file (e.g. DCD).
        pos : str
            Mutation position label (e.g. "3_7").
        i : int
            Iteration index.
        output_path : str
            Directory where the representative structure and plots will be saved.
        logger : Any
            Logger for logging information and errors.
        selection : str
            Atom selection string for alignment and RMSD calculation (default: "backbone").
        eps : float
            DBSCAN eps parameter (default: 1.0).
        min_samples : int
            DBSCAN min_samples parameter (default: 5).
        plot : bool
            Whether to plot the RMSD matrix with clusters (default: True).
        subsample_rate : int
            Subsampling rate for frames to reduce computational load (default: 1000, i.e. use every 1000th frame).  

        Returns
        -------
        representative_frame : int
            Frame index in the original trajectory (not subsampled index).
        largest_cluster_label : int
            DBSCAN label of the largest cluster.
        cluster_counts : Dict[int, int]
            Counts per cluster label.
        
        Raises
        ------
        ValueError
            If subsample_rate is not greater than 0, or if no frames are available after subsampling.
        ValueError
            If the trajectory file cannot be read or if the selection is invalid.
    """
    topology_file = write_topology(
        trajectory_file,
        f"{output_path}/topology_mut_{pos}_{i}.pdb",
        logger=logger,
    )

    u = mda.Universe(topology_file, trajectory_file)

    logger.info("Selection: %s", selection)
    atom_selection = u.select_atoms(selection)

    logger.info("Aligning trajectory to first frame (in memory)")
    align.AlignTraj(u, u, select=selection, in_memory=True).run()

    if subsample_rate <= 0:
        raise ValueError(f"subsample_rate must be > 0 (got {subsample_rate})")

    subsampled_frames = list(range(0, len(u.trajectory), subsample_rate))
    n = len(subsampled_frames)
    if n == 0:
        raise ValueError("No frames available for clustering (empty trajectory).")
    if n == 1:
        # Trivial case: only one subsampled frame
        representative_frame = subsampled_frames[0]
        out_pdb = f"{output_path}/rep_mut_{pos}_{i}.pdb"
        u.trajectory[representative_frame]
        u.atoms.write(out_pdb)
        logger.info("Only one frame available; representative saved: %s", out_pdb)
        return representative_frame, 0, {0: 1}

    logger.info("Subsampling frames: %d total (every %d)", n, subsample_rate)
    logger.info("Computing RMSD matrix (%dx%d)", n, n)

    rmsd_matrix = np.zeros((n, n), dtype=float)
    for idx_i, frame_i in enumerate(subsampled_frames):
        u.trajectory[frame_i]
        ref_positions = atom_selection.positions.copy()

        for idx_j, frame_j in enumerate(subsampled_frames):
            if frame_j >= frame_i:
                u.trajectory[frame_j]
                val = rms.rmsd(ref_positions, atom_selection.positions)
                rmsd_matrix[idx_i, idx_j] = val
                rmsd_matrix[idx_j, idx_i] = val

    logger.info("Running DBSCAN (eps=%.3f, min_samples=%d)", eps, min_samples)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(rmsd_matrix)
    labels = db.labels_

    # Plot if requested
    if plot:
        plot_path = f"{output_path}/rmsd_matrix_{pos}_{i}.png"
        sns.heatmap(rmsd_matrix, xticklabels=False, yticklabels=False, cmap="viridis")
        plt.title("RMSD Matrix with Clustering")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("RMSD matrix plot saved: %s", plot_path)

    cluster_counts_counter = Counter(labels)
    cluster_counts: Dict[int, int] = dict(cluster_counts_counter)

    # If DBSCAN labels everything as noise (-1), still pick the first frame as representative
    largest_cluster_label = max(cluster_counts_counter, key=cluster_counts_counter.get)

    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    representative_frame = subsampled_frames[int(largest_cluster_indices[0])]

    out_pdb = f"{output_path}/rep_mut_{pos}_{i}.pdb"
    u.trajectory[representative_frame]
    u.atoms.write(out_pdb)

    logger.info(
        "Representative frame=%d (label=%s, count=%d) saved: %s",
        representative_frame,
        largest_cluster_label,
        cluster_counts_counter[largest_cluster_label],
        out_pdb,
    )

    return representative_frame, int(largest_cluster_label), cluster_counts
   