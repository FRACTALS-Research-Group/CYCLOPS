import os
import sys
import time
import traceback
from typing import Any, Optional, Tuple

import numpy as np
import openmm
from openmm import app, unit, LangevinIntegrator, CustomExternalForce, MonteCarloBarostat
from openmm.app import PDBFile, Simulation, Modeller, DCDReporter, StateDataReporter
from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer

def fix_pdb(pdb_path: str, output_path: str, logger: Any) -> None:
    """
        Fix a PDB file using PDBFixer, adding missing atoms/residues and hydrogens.
        
        Parameters
        ----------
        pdb_path : str
            Path to the input PDB file.
        output_path : str
            Path to save the fixed PDB file.
        logger : Any
            Logger for logging information and errors.
    """
    fixer = PDBFixer(filename=pdb_path)
    original_chains = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21].strip()
                if chain not in original_chains:
                    original_chains.append(chain)

    try:
        fixer.findMissingResidues()
    except Exception as e:
        logger.warning("findMissingResidues failed for %s: %s", pdb_path, e)
        logger.debug(traceback.format_exc())

    try:
        fixer.findMissingAtoms()
    except Exception as e:
        logger.warning("findMissingAtoms failed for %s: %s", pdb_path, e)
        logger.debug(traceback.format_exc())

    try:
        fixer.addMissingAtoms()
    except Exception as e:
        logger.warning("addMissingAtoms failed for %s: %s", pdb_path, e)
        logger.debug(traceback.format_exc())

    try:
        fixer.findNonstandardResidues()
    except Exception as e:
        logger.warning("findNonstandardResidues failed for %s: %s", pdb_path, e)
        logger.debug(traceback.format_exc())

    try:
        fixer.replaceNonstandardResidues()
    except Exception as e:
        logger.warning("replaceNonstandardResidues failed for %s: %s", pdb_path, e)
        logger.debug(traceback.format_exc())

    try:
        fixer.addMissingHydrogens(7.4)
    except Exception as e:
        logger.warning("addMissingHydrogens failed for %s: %s", pdb_path, e)
        logger.debug(traceback.format_exc())

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # with open(output_path, "w", encoding="utf-8") as out_file:
    #     PDBFile.writeFile(fixer.topology, fixer.positions, out_file)
    
    for chain, old_id in zip(fixer.topology.chains(), original_chains):
        chain.id = old_id
    
    with open(output_path, "w", encoding="utf-8") as out_file:
        PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            out_file,
            keepIds=True
        )

    logger.info("Fixed PDB saved: %s -> %s", pdb_path, output_path)

def simulate_complex(
    protein_pdb_path: str,
    ligand_pdb_path: str,
    base_out: str,
    *,
    logger: Any,
    output_base: str = "output",
    steps: int = 5000,
    step_size: float = 0.002,
    friction_coeff: float = 1.0,
    interval: int = 1000,
    solvate: bool = False,
    padding: float = 10.0,
    water_model: str = "tip3p",
    positive_ion: str = "Na+",
    negative_ion: str = "Cl-",
    ionic_strength: float = 0.15,
    no_neutralize: bool = False,
    equilibration_steps: int = 200,
    protein_force_field: str = "amber/ff14SB.xml",
    water_force_field: str = "amber/tip3p_standard.xml",
    annealing: bool = False,
    temp_start: float = 1000.0,
    temp_end: float = 300.0,
    restrain_protein: bool = True,
    restraint_force: float = 5.0,
    T_restrain: float = 400.0,
    platform: Optional[openmm.Platform] = None,
    checkpoint_interval: int = 50000,
    checkpoint_file: str = "simulation.chk",
) -> Tuple[Simulation, openmm.System, SystemGenerator]:
    """
        Simulate a protein-ligand complex using OpenMM, with optional solvation, annealing, and restraints.
        
        Parameters
        ----------
        protein_pdb_path : str
            Path to the input protein PDB file.
        ligand_pdb_path : str
            Path to the input ligand PDB file.
        base_out : str
            Base name for output files (without extension).
        output_base : str, optional
            Base name for output files (without extension), by default "output".
        steps : int, optional
            Number of production steps to run, by default 5000.
        step_size : float, optional
            Time step size in picoseconds, by default 0.002.
        friction_coeff : float, optional
            Friction coefficient for Langevin dynamics, by default 1.0.
        interval : int, optional
            Interval for reporting and saving trajectory frames, by default 1000.
        solvate : bool, optional
            Whether to add solvent and ions, by default False.
        padding : float, optional
            Padding distance in Å for solvation box, by default 10.0.
        water_model : str, optional
            Water model to use for solvation, by default "tip3p".
        positive_ion : str, optional
            Positive ion type for solvation (e.g. "Na+"), by default "Na+".
        negative_ion : str, optional
            Negative ion type for solvation (e.g. "Cl-"), by default "Cl-".
        ionic_strength : float, optional
            Ionic strength in molar for solvation, by default 0.15.
        no_neutralize : bool, optional
            If True, do not neutralize the system when adding solvent, by default False.
        equilibration_steps : int, optional
            Number of steps to run for equilibration before production, by default 200.
        protein_force_field : str, optional
            Force field XML file for the protein, by default "amber/ff14SB.xml".
        water_force_field : str, optional
            Force field XML file for water, by default "amber/tip3p_standard.xml".
        annealing : bool, optional
            Whether to perform simulated annealing, by default False.
        temp_start : float, optional
            Starting temperature in Kelvin for annealing, by default 1000.0.
        temp_end : float, optional
            Ending temperature in Kelvin for annealing, by default 300.0.
        restrain_protein : bool, optional
            Whether to apply positional restraints to the protein backbone, by default True.
        restraint_force : float, optional
            Force constant in kcal/mol/Å^2 for protein restraints, by default 5.0.
        T_restrain : float, optional
            Temperature in Kelvin above which to turn off restraints during annealing, by default 400.
        platform : openmm.Platform, optional
            OpenMM platform to use for simulation (e.g. CUDA, OpenCL), by default None (auto-detect).
        checkpoint_interval : int, optional
            Interval in steps for saving checkpoints, by default 50000.
        checkpoint_file : str, optional
            Path for saving checkpoint files, by default "simulation.chk".
    
        Returns
        -------
            Tuple[Simulation, openmm.System, SystemGenerator]
                The OpenMM Simulation object, the System, and the SystemGenerator used for the simulation.
    """
    t0 = time.time()

    # --- Output files ---
    output_complex = output_base + "_complex.pdb"
    output_traj_dcd = output_base + "_traj.dcd"
    output_min = output_base + "_minimised.pdb"

    # --- Fix PDBs ---
    fixed_protein_pdb = output_base + "_fixed_protein.pdb"
    fixed_ligand_pdb = output_base + "_fixed_ligand.pdb"

    logger.info("Fixing protein PDB...")
    fix_pdb(protein_pdb_path, fixed_protein_pdb, logger=logger)

    logger.info("Fixing peptide ligand PDB...")
    fix_pdb(ligand_pdb_path, fixed_ligand_pdb, logger=logger)

    # --- Read protein & ligand ---
    protein_pdb = PDBFile(fixed_protein_pdb)
    ligand_pdb = PDBFile(fixed_ligand_pdb)

    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    modeller.add(ligand_pdb.topology, ligand_pdb.getPositions())
    logger.info("System atoms after adding ligand: %d", modeller.topology.getNumAtoms())

    # --- SystemGenerator ---
    forcefield_kwargs = {
        "constraints": app.HBonds,
        "rigidWater": True,
        "removeCMMotion": False,
        "hydrogenMass": 4 * unit.amu,
    }
    system_generator = SystemGenerator(
        forcefields=[protein_force_field, water_force_field],
        forcefield_kwargs=forcefield_kwargs,
    )

    # --- Solvation ---
    temperature_unit = float(temp_end) * unit.kelvin
    if solvate:
        logger.info("Adding solvent (model=%s, padding=%s Å, ionic_strength=%s M)", water_model, padding, ionic_strength)
        modeller.addSolvent(
            system_generator.forcefield,
            model=water_model,
            padding=float(padding) * unit.angstroms,
            positiveIon=positive_ion,
            negativeIon=negative_ion,
            ionicStrength=float(ionic_strength) * unit.molar,
            neutralize=not no_neutralize,
        )
        logger.info("System atoms after solvation: %d", modeller.topology.getNumAtoms())

    # --- Save complex PDB ---
    with open(output_complex, "w", encoding="utf-8") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    logger.info("Complex PDB saved: %s", output_complex)

    # --- Create System ---
    system = system_generator.create_system(modeller.topology)

    if solvate:
        system.addForce(MonteCarloBarostat(1 * unit.atmospheres, temperature_unit, 25))
        logger.info("Barostat enabled (1 atm).")

    # --- Restraints ---
    restraint_force_obj: Optional[CustomExternalForce] = None
    if restrain_protein:
        restraint_force_obj = CustomExternalForce("scale*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        restraint_force_obj.addGlobalParameter("scale", 0.0)
        restraint_force_obj.addPerParticleParameter("x0")
        restraint_force_obj.addPerParticleParameter("y0")
        restraint_force_obj.addPerParticleParameter("z0")

        count = 0
        for atom in modeller.topology.atoms():
            if atom.residue.chain.id == "A" and atom.name in ["N", "CA", "C"]:
                pos = modeller.positions[atom.index]
                restraint_force_obj.addParticle(atom.index, [pos.x, pos.y, pos.z])
                count += 1

        system.addForce(restraint_force_obj)
        logger.info("Backbone restraints added for %d atoms (chain A, N/CA/C).", count)

    # --- Integrator & Simulation ---
    integrator = LangevinIntegrator(
        temperature_unit,
        float(friction_coeff) / unit.picosecond,
        float(step_size) * unit.picoseconds,
    )

    if platform is None:
        platform = openmm.Platform.getPlatformByName("CUDA")
        logger.info("Using platform: CUDA")

    simulation = Simulation(modeller.topology, system, integrator, platform=platform)
    simulation.context.setPositions(modeller.positions)

    # --- Minimize ---
    logger.info("Minimising ...")
    simulation.minimizeEnergy()

    with open(output_min, "w", encoding="utf-8") as f:
        PDBFile.writeFile(
            modeller.topology,
            simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
            f,
            keepIds=True,
        )
    logger.info("Minimised PDB saved: %s", output_min)

    # --- Equilibrate ---
    simulation.context.setVelocitiesToTemperature(temperature_unit)
    logger.info("Equilibrating (%d steps) ...", int(equilibration_steps))
    simulation.step(int(equilibration_steps))

    # --- Reporters ---
    os.makedirs(base_out, exist_ok=True)
    simulation.reporters.append(DCDReporter(output_traj_dcd, int(interval), enforcePeriodicBox=True))
    simulation.reporters.append(
        StateDataReporter(sys.stdout, int(interval) * 5, step=True, potentialEnergy=True, temperature=True)
    )
    simulation.reporters.append(
        StateDataReporter(
            os.path.join(base_out, "md_log.txt"),
            int(interval) * 5,
            step=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
        )
    )

    # --- Production / Annealing ---
    if annealing:
        logger.info("Starting simulated annealing ...")
        annealing_steps = int(steps) // 4
        production_steps = int(steps) - annealing_steps

        temp_array = np.linspace(float(temp_start), float(temp_end), annealing_steps)
        for idx, T in enumerate(temp_array):
            simulation.context.setVelocitiesToTemperature(float(T) * unit.kelvin)
            if restrain_protein and restraint_force_obj is not None:
                simulation.context.setParameter("scale", float(restraint_force) if float(T) > float(T_restrain) else 0.0)

            simulation.step(1)

            if idx % 1000 == 0:
                logger.info("Annealing step %d/%d, T=%.1f K", idx, annealing_steps, float(T))

        logger.info("Annealing complete. Starting production run (%d steps) at %.1f K ...", production_steps, float(temp_end))
        integrator.setTemperature(float(temp_end) * unit.kelvin)
        simulation.step(int(production_steps))
    else:
        logger.info("Starting production run (%d steps) ...", int(steps))
        simulation.step(int(steps))

    elapsed_min = (time.time() - t0) / 60.0
    logger.info("Simulation complete in %.2f min", elapsed_min)

    return simulation, system, system_generator
   