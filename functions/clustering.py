import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import MDAnalysis as mda

def write_topology(input_pdb, output_pdb):
    """
    Write the topology of a given PDB file to a new PDB file.
    
    Parameters:
    input_pdb (str): Path to the input PDB file.
    output_pdb (str): Path to the output PDB file.
    """
    # Load the PDB file into an MDAnalysis Universe
    u = mda.Universe(input_pdb)

    # Select the first frame
    u.trajectory[0]

    # Write the first frame to a new PDB file
    with mda.Writer(output_pdb, multiframe=False) as W:
        W.write(u)
    return output_pdb

    
def get_most_probable_conformation(trajectory_file, pos, i, output_path, selection='backbone', eps=1.0, min_samples=5, plot=True, subsample_rate=1000):
    # Load the trajectory and the topology
    topology_file = write_topology(trajectory_file, f"{output_path}/topology_mut_{pos}_{i}.pdb")
    u = mda.Universe(topology_file, trajectory_file)
    print("Performing selection")
    # Select the atoms for alignment, usually backbone atoms
    atom_selection = u.select_atoms(selection)
    
    print("Performing alignment")
    # Align the trajectory to the first frame
    align.AlignTraj(u, u, select=selection, in_memory=True).run()
    
    print("Subsampling frames")
    # Subsample the frames
    subsampled_frames = range(0, len(u.trajectory), subsample_rate)
    
    print("Computing RMSD")
    # Calculate the RMSD matrix for the subsampled frames
    n_subsampled_frames = len(subsampled_frames)
    rmsd_matrix = np.zeros((n_subsampled_frames, n_subsampled_frames))
    
    for idx_i, x in enumerate(subsampled_frames):
        u.trajectory[x]
        ref_positions = atom_selection.positions.copy()
        for idx_j, j in enumerate(subsampled_frames):
            if j >= x:
                u.trajectory[j]
                rmsd_matrix[idx_i, idx_j] = rms.rmsd(ref_positions, atom_selection.positions)
                rmsd_matrix[idx_j, idx_i] = rmsd_matrix[idx_i, idx_j]
    
    # Perform DBSCAN clustering
    print("Performing DBSCAN")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(rmsd_matrix)
    
    # Retrieve the cluster labels
    labels = db.labels_
    
    # Plot the RMSD matrix with clusters
    if plot:
        sns.heatmap(rmsd_matrix, xticklabels=False, yticklabels=False, cmap='viridis')
        plt.title('RMSD Matrix with Clustering')
        plt.savefig("rmsd.png")
    
    # Determine the most populated cluster
    cluster_counts = Counter(labels)
    largest_cluster_label = max(cluster_counts, key=cluster_counts.get)
    
    # Get representative structure from the largest cluster
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    representative_frame = subsampled_frames[largest_cluster_indices[0]]
    
    # Save or retrieve the most probable conformation
    u.trajectory[representative_frame]
    u.atoms.write(f'{output_path}/rep_mut_{pos}_{i}.pdb')
    print(f'The most probable conformation is saved as {output_path}/rep_mut_{pos}_{i}.pdb')
    
    return representative_frame, largest_cluster_label, cluster_counts

# # Example usage
# u = mda.Universe("ref_lig.pdb")

# # Select the first frame
# u.trajectory[0]

# # Write the first frame to a new PDB file
# with mda.Writer("topology.pdb", multiframe=False) as W:
#     W.write(u)

# Run the analysis with subsampling every 100th frame
# representative_frame, largest_cluster_label, cluster_counts = get_most_probable_conformation('topology.pdb', 'ref_lig.pdb', subsample_rate=500)

