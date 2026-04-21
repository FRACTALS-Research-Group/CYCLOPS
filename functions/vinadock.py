from vina import Vina
import numpy as np

# Given the coordinates in a PDBQT file, return the geometric center 
# --> define the center of the binding pocket (file = ref_ligand)
# --> define the center of new ligands
# --> translate the ligand (see change_pdbqt())
def parse_pdbqt(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                coo_line = line.split()
                x = float(coo_line[6])
                y = float(coo_line[7])
                z = float(coo_line[8])
                coordinates.append((x, y, z))
    coordinates = np.array(coordinates)
    min_coord = np.min(coordinates, axis=0)
    max_coord = np.max(coordinates, axis=0)
    center = (min_coord + max_coord) / 2
    return center


def change_pdbqt(ref_path, new_path):
    # Definizione del centro geometrico di box e ligando
    box_center = parse_pdbqt(ref_path)
    print("Center of the box set at: ", box_center)
    lig_center = parse_pdbqt(new_path)
    print("Center of the ligand set at: ", lig_center)
    # Calcolo del vettore di traslazione per ogni punto
    v_trans = lig_center - box_center
    print("Translation vector: ", v_trans)
    dx, dy, dz = v_trans
    
    with open(new_path, 'r') as file:
        lines = file.readlines()
    with open(new_path, 'w') as file:
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                coo_line = line.split()
                x = float(coo_line[6]) - dx
                y = float(coo_line[7]) - dy
                z = float(coo_line[8]) - dz

                # Reformat the line with the new coordinates
                new_line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                file.write(new_line)
            else:
                file.write(line)
    return box_center



def parse_affinity(file_path):
    affinities = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("REMARK VINA RESULT:"):
                affinity = float(line[22:29])
                affinities.append(affinity)
        mean = sum(affinities)/len(affinities)
        best = min(affinities)
    return best, mean



def vina_dock(
    receptor_path, ref_ligand_path, new_ligand_path, 
    minimized_output_path, docking_vina_out_path, dock=True,
    box_size=[20,20,20], exhaustiveness=32, n_poses=10):
    
    v = Vina(sf_name='vina')

    # Set the receptor
    v.set_receptor(receptor_path)
    print("Receptor set")
    
    # Set the center of the box and put the ligand inside it
    box_center = change_pdbqt(ref_ligand_path, new_ligand_path)
    
    v.set_ligand_from_file(new_ligand_path)
    v.compute_vina_maps(center=box_center, box_size=box_size)

    try:
        energy = v.score()[0]
        print('Score before minimization: %.3f (kcal/mol)' % energy)
    except:
        v.compute_vina_maps(center=box_center, box_size=[30,30,30])
        energy = v.score()[0]
        print('Score before minimization: %.3f (kcal/mol)' % energy)
        
    energy_minimized = v.optimize()[0]
    print('Score after minimization : %.3f (kcal/mol)' % energy_minimized)
    v.write_pose(minimized_output_path, overwrite=True)

    if dock:
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        v.write_poses(docking_vina_out_path, n_poses=n_poses, overwrite=True)
    
        poses_best_energy, poses_mean_energy = parse_affinity(docking_vina_out_path)
        
    else:
        poses_best_energy = energy_minimized
        poses_mean_energy = energy_minimized
    
    return poses_best_energy, poses_mean_energy



def calculate_and_format_docking_scores(mean_docking_score, best_docking_score):
    absolute_error = abs(mean_docking_score - best_docking_score)
    relative_error = (absolute_error / mean_docking_score) * 100 if mean_docking_score != 0 else float('inf')
    
    docking_score_with_absolute_error = f"{best_docking_score:.3f} ± {absolute_error:.3f} kcal/mol"
    docking_score_with_relative_error = f"{best_docking_score:.3f} ± {relative_error:.2f}%"
    
    return docking_score_with_absolute_error, docking_score_with_relative_error