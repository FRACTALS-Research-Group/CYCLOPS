from Bio import PDB

def cif2pdb(input_cif, output_pdb):
    # Parser per file mmCIF
    parser = PDB.MMCIFParser(QUIET=True)
    
    # Legge il file mmCIF e crea un oggetto struttura
    structure = parser.get_structure("AF3_Model", input_cif)
    
    # Scrive la struttura in formato PDB
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)

import os
from glob import glob
import subprocess

def pdb2pdbqt_rece(input_folder, output_folder, base, conversion='*.pdb'):
    os.makedirs(output_folder, exist_ok=True)
    pdb_files = glob(os.path.join(input_folder, conversion))
    for pdb_file in pdb_files:
        pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]  # toglie .pdb
        pdbqt_file = os.path.join(output_folder, f"{pdb_name}.pdbqt")
        # pdbqt_file = os.path.join(output_folder, f"{os.path.basename(pdb_file)}.pdbqt")
        command = [f'{base}/functions/ADFRsuite-1.1dev/ADFRsuite_x86_64Linux_1.1dev/docking/bin/prepare_receptor',
                   '-r', pdb_file, '-o', pdbqt_file]
        subprocess.call(command)
        print(f"Convertito: {pdb_file} → {pdbqt_file}")
    else:
        pdb_file = f"{input_folder}.pdb"
        pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]  # toglie .pdb
        pdbqt_file = os.path.join(output_folder, f"{pdb_name}.pdbqt")
        # pdbqt_file = os.path.join(output_folder, f"{os.path.basename(pdb_file)}.pdbqt")
        command = [f'{base}/functions/ADFRsuite-1.1dev/ADFRsuite_x86_64Linux_1.1dev/docking/bin/prepare_receptor',
                   '-r', pdb_file, '-o', pdbqt_file]
        subprocess.call(command)
        print(f"Convertito: {pdb_file} → {pdbqt_file}")
        # print(f"Il file {pdb_files} non esiste.")


def pdb2pdbqt_lig(input_folder, output_folder, base, conversion="*.pdb"):
    os.makedirs(output_folder, exist_ok=True)

    # pdb_file = os.path.join(input_folder, conversion)
    pdb_file = f"{input_folder}.pdb"
    if os.path.exists(pdb_file):
        pdb_name = os.path.splitext(os.path.basename(pdb_file))[0]  # toglie .pdb
        # pdb_name = os.path.join(input_folder, f"{pdb_name}.pdb")
        pdbqt_file = os.path.join(output_folder, f"{pdb_name}.pdbqt")

        command = [
            f"{base}/functions/ADFRsuite-1.1dev/ADFRsuite_x86_64Linux_1.1dev/docking/bin/prepare_ligand",
            "-l", pdb_file,
            "-o", pdbqt_file
        ]
        subprocess.call(command)
        print(f"Convertito: {pdb_file} → {pdbqt_file}")
    else:
        print(f"Il file {pdb_file} non esiste.")

# def pdbqt2pdb(input_folder, output_folder, base, pos, i):
#     os.makedirs(output_folder, exist_ok=True)

#     pdbqt_files = glob(os.path.join(input_folder, f'out_{pos}_{i}.pdbqt'))
#     if not pdbqt_files:
#         print(f"[ERRORE] Nessun file trovato in {input_folder} con nome out_{pos}_{i}.pdbqt")
#         return

#     for pdbqt_file in pdbqt_files:
#         # Costruisci nome output .pdb
#         pdb_file = os.path.join(
#             output_folder,
#             os.path.basename(pdbqt_file).replace('.pdbqt', '.pdb')
#         )

#         # Controllo: deve terminare con .pdb
#         if not pdb_file.endswith(".pdb"):
#             raise ValueError(f"Output non valido: {pdb_file}")

#         # Comando obabel
#         command = [
#             f'{base}/functions/ADFRsuite-1.1dev/ADFRsuite_x86_64Linux_1.1dev/docking/bin/obabel',
#             '-ipdbqt', pdbqt_file,
#             '-O', pdb_file
#         ]

#         print("[DEBUG] Input:", pdbqt_file)
#         print("[DEBUG] Output:", pdb_file)
#         print("[DEBUG] Command:", " ".join(command))

#         # Esegui
#         ret = subprocess.call(command)
#         if ret != 0:
#             print(f"[ERRORE] obabel ha restituito codice {ret}")
#         else:
#             print(f"[OK] Creato {pdb_file}")

def pdbqt2pdb(input_folder, output_folder, base, pos, i):
    os.makedirs(output_folder, exist_ok=True)

    pdbqt_files = glob(os.path.join(input_folder, f'out_{pos}_{i}.pdbqt'))
    if not pdbqt_files:
        print(f"[ERRORE] Nessun file trovato in {input_folder} con nome out_{pos}_{i}.pdbqt")
        return

    for pdbqt_file in pdbqt_files:
        # --- Filtro: tieni solo MODEL 1 ---
        with open(pdbqt_file, "r") as f:
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
                    # stop writing da MODEL 2 in poi
                    in_first_model = False
                    continue

            if line.startswith("ENDROOT"):
                in_section_to_remove = True

            if not in_section_to_remove and in_first_model:
                new_lines.append(line)

            if line.startswith("ENDMDL"):
                in_section_to_remove = False

        # Sovrascrivi il file .pdbqt con la versione ridotta
        with open(pdbqt_file, "w") as f:
            f.writelines(new_lines)

        # --- Conversione con obabel ---
        pdb_file = os.path.join(
            output_folder,
            os.path.basename(pdbqt_file).replace('.pdbqt', '.pdb')
        )

        if not pdb_file.endswith(".pdb"):
            raise ValueError(f"Output non valido: {pdb_file}")

        command = [
            f'{base}/functions/ADFRsuite-1.1dev/ADFRsuite_x86_64Linux_1.1dev/docking/bin/obabel',
            '-ipdbqt', pdbqt_file,
            '-O', pdb_file
        ]

        ret = subprocess.call(command)
        if ret != 0:
            print(f"[ERRORE] obabel ha restituito codice {ret}")
        else:
            print(f"[OK] Creato {pdb_file}")
            
            
# pdb2pdbqt_lig('docking/inputs/seq_3_0/rep_lig_3_0', 'docking/inputs/seq_3_0', ".", conversion=f'rep_lig_3_0.pdb')