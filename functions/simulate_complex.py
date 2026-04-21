import sys, time
from openmmforcefields.generators import SystemGenerator
import openmm
from openmm import app, unit, LangevinIntegrator, CustomExternalForce, MonteCarloBarostat
from openmm.app import PDBFile, Simulation, Modeller, DCDReporter, StateDataReporter
from pdbfixer import PDBFixer
from openmm.app import PDBxFile
# import functions.utils as utils
import numpy as np

import traceback
def fix_pdb(pdb_path, output_path):
    """
    Fixes a PDB file using PDBFixer, adding missing atoms and hydrogens.

    Args:
        pdb_path (str): Path to the input PDB file.
        output_path (str): Path to save the fixed PDB file.
        is_peptide (bool): If True, treat the molecule as a peptide; otherwise, treat as a protein.
    """
    fixer = PDBFixer(filename=pdb_path)
    try: fixer.findMissingResidues()
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()

    try: fixer.findMissingAtoms()
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
    try: fixer.addMissingAtoms()
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
    try: fixer.findNonstandardResidues()
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
    try: fixer.replaceNonstandardResidues()
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
    try: fixer.addMissingHydrogens(7.4)
    except Exception as e:
        print(f"Error during execution: {e}")
        traceback.print_exc()
    
    # Save the fixed PDB to the specified output path
    with open(output_path, 'w') as out_file:
        PDBFile.writeFile(fixer.topology, fixer.positions, out_file)
    print(f'Fixed PDB saved to {output_path}')
    

def replace_last_chain_identifier(pdb_file, new_chain_id):
    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    # Find all unique chain identifiers
    chain_ids = set()
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            chain_ids.add(line[21])

    # Determine the last chain ID
    last_chain_id = None
    for line in reversed(lines):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            last_chain_id = line[21]
            break
    atoms = range(3585,3920)
    # Replace the last chain ID if it is 'A'
    if last_chain_id == 'A':
        lines = [line if not (line.startswith("ATOM") or line.startswith("HETATM")) or line[21] != 'A' or int(line[7:11]) not in atoms else line[:21] + new_chain_id + line[22:] for line in lines]

    # Write the modified lines to a new file
    with open(f"{pdb_file}", 'w') as file:
        file.writelines(lines)
        

import time, os, sys
import numpy as np
from openmm import unit, LangevinIntegrator, CustomExternalForce, MonteCarloBarostat
from openmm import app
from openmm.app import PDBFile, Modeller, DCDReporter, StateDataReporter, Simulation, CheckpointReporter
from openmmforcefields.generators import SystemGenerator

def simulate_complex(protein_pdb_path, ligand_pdb_path, base_out, 
                     output_base='output', steps=5000, step_size=0.002,
                     friction_coeff=1, interval=1000, solvate=False, padding=10,
                     water_model="tip3p", positive_ion="Na+", negative_ion="Cl-", ionic_strength=0.15,
                     no_neutralize=False, equilibration_steps=200,
                     protein_force_field='amber/ff14SB.xml', water_force_field='amber/tip3p_standard.xml',
                     annealing=False, temp_start=1000, temp_end=300,
                     restrain_protein=True, restraint_force=5.0, T_restrain=400,
                     platform=None, checkpoint_interval=50000, checkpoint_file="simulation.chk"):

    t0 = time.time()

    # --- Output files ---
    output_complex = output_base + '_complex.pdb'
    output_traj_dcd = output_base + '_traj.dcd'
    output_min = output_base + '_minimised.pdb'

    # --- Fix PDBs ---
    fixed_protein_pdb = output_base + '_fixed_protein.pdb'
    fixed_ligand_pdb = output_base + '_fixed_ligand.pdb'
    print('Fixing protein PDB...')
    fix_pdb(protein_pdb_path, fixed_protein_pdb)
    print('Fixing peptide ligand PDB...')
    fix_pdb(ligand_pdb_path, fixed_ligand_pdb)
    
    # print("Replacing ligand chain identifier to avoid conflict...")
    # replace_last_chain_identifier(fixed_ligand_pdb, 'C')

    # --- Read protein & ligand ---
    protein_pdb = PDBFile(fixed_protein_pdb)
    ligand_pdb = PDBFile(fixed_ligand_pdb)
    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    modeller.add(ligand_pdb.topology, ligand_pdb.getPositions())
    print(f'System has {modeller.topology.getNumAtoms()} atoms after adding ligand')

    # --- Create SystemGenerator and System (keep alive until end) ---
    forcefield_kwargs = {'constraints': app.HBonds, 'rigidWater': True,
                         'removeCMMotion': False, 'hydrogenMass': 4 * unit.amu}
    system_generator = SystemGenerator(forcefields=[protein_force_field, water_force_field],
                                       forcefield_kwargs=forcefield_kwargs)

    # --- Add solvent if requested ---
    if solvate:
        print('Adding solvent...')
        modeller.addSolvent(system_generator.forcefield, model=water_model,
                            padding=padding * unit.angstroms,
                            positiveIon=positive_ion, negativeIon=negative_ion,
                            ionicStrength=ionic_strength * unit.molar,
                            neutralize=not no_neutralize)
        print(f'System has {modeller.topology.getNumAtoms()} atoms after solvation')

    # --- Save complex PDB ---
    with open(output_complex, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

    # --- Create System ---
    system = system_generator.create_system(modeller.topology)

    # --- Add barostat if solvated ---
    temperature_unit = temp_end * unit.kelvin
    if solvate:
        system.addForce(MonteCarloBarostat(1 * unit.atmospheres, temperature_unit, 25))

    # --- Apply backbone restraints ---
    restraint_force_obj = None
    if restrain_protein:
        restraint_force_obj = CustomExternalForce(
            "scale*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        )
        restraint_force_obj.addGlobalParameter("scale", 0.0)
        restraint_force_obj.addPerParticleParameter("x0")
        restraint_force_obj.addPerParticleParameter("y0")
        restraint_force_obj.addPerParticleParameter("z0")
        for atom in modeller.topology.atoms():
            if atom.residue.chain.id == 'A' and atom.name in ['N','CA','C']:
                pos = modeller.positions[atom.index]
                restraint_force_obj.addParticle(atom.index, [pos.x, pos.y, pos.z])
        system.addForce(restraint_force_obj)

    # --- Integrator & Simulation ---
    integrator = LangevinIntegrator(temperature_unit, friction_coeff / unit.picosecond,
                                    step_size * unit.picoseconds)
    if platform is None:
        platform = openmm.Platform.getPlatformByName('CUDA')  # or CUDA if available
    simulation = Simulation(modeller.topology, system, integrator, platform=platform)
    simulation.context.setPositions(modeller.positions)

    # --- Minimize ---
    print('Minimising ...')
    simulation.minimizeEnergy()
    with open(output_min, 'w') as f:
        PDBFile.writeFile(modeller.topology,
                          simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
                          f, keepIds=True)

    # --- Equilibrate ---
    simulation.context.setVelocitiesToTemperature(temperature_unit)
    print('Equilibrating ...')
    simulation.step(equilibration_steps)

    # --- Add reporters ---
    simulation.reporters.append(DCDReporter(output_traj_dcd, interval, enforcePeriodicBox=True))
    simulation.reporters.append(StateDataReporter(sys.stdout, interval*5, step=True,
                                                  potentialEnergy=True, temperature=True))
    simulation.reporters.append(StateDataReporter(os.path.join(base_out, "md_log.txt"), interval*5,
                                                  step=True, potentialEnergy=True, temperature=True, volume=True))
    
    # # --- Checkpoint reporter ---
    # simulation.reporters.append(CheckpointReporter(checkpoint_file, checkpoint_interval))

    # # --- Ripartenza dal checkpoint ---
    # if os.path.exists(checkpoint_file):
    #     print(f"\n>>> Checkpoint trovato, riprendo da {checkpoint_file}\n")
    #     simulation.loadCheckpoint(checkpoint_file)
    # else:
    #     print("\n>>> Nessun checkpoint, inizio nuova simulazione\n")
    #     simulation.context.setPositions(modeller.positions)
    #     simulation.context.setVelocitiesToTemperature(temp_end*unit.kelvin)



    # --- Production / Annealing ---
    if annealing:
        print("Starting simulated annealing ...")
        annealing_steps = steps // 4
        production_steps = steps - annealing_steps
        temp_array = np.linspace(temp_start, temp_end, annealing_steps)
        for i, T in enumerate(temp_array):
            simulation.context.setVelocitiesToTemperature(T * unit.kelvin)
            if restrain_protein and restraint_force_obj is not None:
                simulation.context.setParameter("scale", restraint_force if T > T_restrain else 0.0)
            simulation.step(1)
            if i % 1000 == 0:
                print(f"Annealing step {i}/{annealing_steps}, T={T:.1f} K")
        print(f"Annealing complete. Starting production run ({production_steps} steps) at {temp_end} K ...")
        integrator.setTemperature(temp_end * unit.kelvin)
        simulation.step(production_steps)
    else:
        print('Starting production run ...')
        simulation.step(steps)

    print(f"Simulation complete in {(time.time()-t0)/60:.2f} min")

    # --- Return Simulation objects if needed (kept alive) ---
    return simulation, system, system_generator



# def parse_pdb(file_path):
#     coords = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             if line.startswith(("ATOM", "HETATM")):
#                 parts = line.split()
#                 coords.append([float(parts[6]), float(parts[7]), float(parts[8])])
#     coords = np.array(coords)
#     return (coords.min(axis=0) + coords.max(axis=0)) / 2


# def change_pdb(ref_path, new_path):
#     box_center = parse_pdb(ref_path)
#     lig_center = parse_pdb(new_path)
#     v_trans = lig_center - box_center
#     print(f"Box center: {box_center}, Ligand center: {lig_center}, Translation vector: {v_trans}")

#     dx, dy, dz = v_trans
#     with open(new_path, 'r') as f:
#         lines = f.readlines()
#     with open(new_path, 'w') as f:
#         for line in lines:
#             if line.startswith(("ATOM", "HETATM")):
#                 parts = line.split()
#                 x, y, z = float(parts[6]) - dx, float(parts[7]) - dy, float(parts[8]) - dz
#                 f.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
#             else:
#                 f.write(line)


# change_pdb("end.pdb", "inputs/ref_1AKJ/pdb/rep_lig_ref_0.pdb")

# simulate_complex(protein_pdb_path="inputs/ref_1AKJ/pdb/rep_rece_ref_0.pdb", ligand_pdb_path="end.pdb", 
#                     base_out=".",
#                      output_base='output', steps=50000, step_size=0.002,
#                      friction_coeff=1, interval=1000, solvate=False, padding=10,
#                      water_model="tip3p", positive_ion="Na+", negative_ion="Cl-", ionic_strength=0.15,
#                      no_neutralize=False, equilibration_steps=200, protein_force_field='amber/ff14SB.xml',
#                      water_force_field='amber/tip3p_standard.xml',
#                      annealing=False, temp_start=1000, temp_end=300,
#                      restrain_protein=True, restraint_force=5.0, T_restrain=400)