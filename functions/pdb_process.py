def split_pdb_by_chain(input_pdb, output_pdb_ab, output_pdb_c):
    """
    Divide un file PDB in due file:
    - Uno contenente le catene A e B
    - Uno contenente la catena C
    """
    with open(input_pdb, 'r') as infile, \
         open(output_pdb_ab, 'w') as out_ab, \
         open(output_pdb_c, 'w') as out_c:
        
        for line in infile:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21]  # La colonna 22 (indice 21) contiene l'ID della catena
                if chain_id in ['A']:
                    out_ab.write(line)
                # elif chain_id == 'C':
                #     out_c.write(line)
                elif chain_id == 'B':
                    out_c.write(line)
            else:
                out_ab.write(line)
                out_c.write(line)


import numpy as np 

def parse_pdb(file_path):
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

def prepare_ligand(ref_path, new_path):
    # Calcolo dei centri geometrici di box e ligando
    box_center = parse_pdb(ref_path)
    print("Center of the box set at: ", box_center)
    lig_center = parse_pdb(new_path)
    print("Center of the ligand set at: ", lig_center)

    # Calcolo del vettore di traslazione
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

