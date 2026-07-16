# CYCLOPS
### Cyclic peptide Ligand Optimization Pipeline for Structure-based Design

> A Monte Carlo optimization pipeline for computational peptide design targeting protein-peptide binding affinity.

---

## Overview

CYCLOPS implements a Metropolis–Hastings Monte Carlo (MC) search over the amino acid sequence space of a target peptide ligand. At each iteration, a candidate sequence is evaluated through a tiered computational funnel:

1. **AlphaFold3** structure prediction of the mutated peptide
2. **AutoDock Vina** pre-screening docking against the receptor
3. **OpenMM** molecular dynamics (MD) simulation of the complex
4. **DBSCAN** clustering to extract the most populated representative conformation
5. **AutoDock Vina** post-MD rescoring on the relaxed complex

Mutations are accepted or rejected according to the Metropolis criterion with an exponential temperature annealing schedule, allowing the algorithm to escape local optima early in the search and converge to high-affinity sequences later.

---

## Computational Workflow

```
┌──────────────────────────────────────────────────────────┐
│                  Monte Carlo Loop (N iterations)          │
│                                                          │
│  1. Propose random amino-acid mutation(s)                │
│        ↓                                                 │
│  2. AlphaFold3 — 3D structure prediction                 │
│        ↓                                                 │
│  3. AutoDock Vina — pre-MD docking                       │
│        ↓ (only if score < −5 kcal/mol)                   │
│  4. OpenMM MD — explicit-solvent simulation              │
│        ↓                                                 │
│  5. DBSCAN clustering — representative frame extraction  │
│        ↓                                                 │
│  6. AutoDock Vina — post-MD rescoring                    │
│        ↓                                                 │
│  7. Metropolis accept / reject                           │
│        ↓                                                 │
│  8. Checkpoint & plot                                    │
└──────────────────────────────────────────────────────────┘
```

### Metropolis Acceptance Criterion

The probability of accepting a mutation is:

$$P_{\text{acc}} = \min\!\left(1,\; e^{-\Delta E / T}\right)$$

where $\Delta E = E_{\text{new}} - E_{\text{old}}$ is the difference in post-MD docking score (kcal/mol) and $T$ is a dimensionless temperature parameter that follows an exponential annealing schedule:

$$T(i) = \max\!\left(T_0 \cdot e^{-\lambda i},\; T_{\min}\right)$$

with decay constant $\lambda = 0.01$ and $T_{\min} = 0.4$.

---

## Repository Structure

```
ALLIANCE/
├── main.py                  # Main MC optimization loop
├── main_new.py              # Refactored pipeline (type-annotated)
├── step_0.py                # Pre-run: reference complex simulation & baseline docking
├── my_env.yaml              # Conda environment specification
└── functions/
    ├── af3.py               # AlphaFold3 runner
    ├── analyse.py           # Legacy analysis utilities
    ├── analysis.py          # MDTraj-based trajectory analysis & reimaging
    ├── checkpoints.py       # Checkpoint save / load (fault tolerance)
    ├── clustering.py        # DBSCAN clustering on RMSD matrix
    ├── json_maker.py        # AlphaFold3 JSON input generator
    ├── mutations.py         # Random amino-acid mutation engine
    ├── pdb_converters.py    # CIF → PDB, PDB → PDBQT converters
    ├── pdb_process.py       # PDB chain splitting utilities
    ├── setup.py             # Argument parsing, path setup, logging
    ├── simulate_complex.py  # OpenMM MD simulation (explicit solvent)
    └── vinadock.py          # AutoDock Vina wrapper & score parser
```

---

## Dependencies

CYCLOPS requires a dedicated conda environment. All dependencies are specified in `my_env.yaml`.

| Component | Package(s) |
|---|---|
| Structure prediction | AlphaFold3 (external installation) |
| Molecular docking | `vina` (AutoDock Vina Python API) |
| Molecular dynamics | `openmm`, `openmmforcefields`, `pdbfixer` |
| Force field | AMBER (via `ambertools`) |
| Trajectory analysis | `mdanalysis`, `mdtraj` |
| Clustering | `scikit-learn` (DBSCAN) |
| Numerics / plotting | `numpy`, `pandas`, `matplotlib`, `seaborn` |
| Structural I/O | `biopython`, `mmcif_pdbx` |

### Installation

```bash
conda env create -f my_env.yaml
conda activate alphafold3
```

AlphaFold3 must be installed separately and its runner script placed at:

```
/opt/miniconda3/envs/alphafold3/alphafold3/run_alphafold.py
```

Refer to the [AlphaFold3 repository](https://github.com/google-deepmind/alphafold3) for installation instructions.

---

## How to Run

The pipeline is split into two stages that must be executed in order.

### Step 0 — Reference system preparation (run once)

`step_0.py` simulates the unmodified receptor–peptide complex, extracts the most representative conformation via DBSCAN clustering, performs a baseline AutoDock Vina docking, and saves the resulting scores to `step0_results.json`. This file is the single source of truth for the initial binding affinity used by the MC loop.

**Prepare input files** — place the receptor PDB and reference ligand PDB in the directory pointed to by `--base_dir` (default `inputs/ref/pdb`):

```
<CYCLOPS_root>/inputs/ref/pdb/
    receptor.pdb
    ligand.pdb
```

**Run step_0:**

```bash
python step_0.py \
    --ligand_pdb       inputs/ref/pdb/ligand.pdb \
    --ref_lig_file     inputs/ref/pdb/ligand.pdb \
    --input_rece_file  inputs/ref/pdb/receptor.pdb \
    --src_route        functions/ \
    --steps            250000 \
    --base_dir         inputs/ref/pdb \
    --box_size         20 20 20 \
    --log_file         step_0.log \
    --results_path     /path/to/CYCLOPS/step0_results.json
```

This writes `step0_results.json` at the path specified by `--results_path`. Use the same absolute path for `--step0_results` in `main.py` (or place the file at `<base_dir>/step0_results.json`, which is the default location that `main.py` expects).

#### `step_0.py` arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--ligand_pdb` | `str` | *(required)* | Path to original reference ligand PDB |
| `--ref_lig_file` | `str` | *(required)* | Path to cyclic peptide PDB |
| `--input_rece_file` | `str` | *(required)* | Path to receptor PDB |
| `--src_route` | `str` | *(required)* | Path to the `functions/` directory |
| `--steps` | `int` | `250000` | MD integration steps for the reference run |
| `--base_dir` | `str` | `inputs/ref/pdb` | Output directory for reference run files |
| `--box_size` | `float float float` | `20 20 20` | Docking box dimensions (Å) |
| `--log_file` | `str` | `step_0.log` | Log file path |
| `--results_path` | `str` | `step0_results.json` | **Destination of the JSON results file read by `main.py`** |

---

### Step 1 — Monte Carlo optimization loop

`main.py` reads `step0_results.json` to initialize the reference docking score and the scores CSV, then runs the full MC search. It must be executed after `step_0.py` has successfully written its results file.

```bash
python main.py \
    --base_dir         /path/to/CYCLOPS \
    --input_rece_file  receptor.pdb \
    --ref_lig_file     ligand.pdb \
    --ref_seq          CAADQTQDTEAAC \
    --keep_pos         0 12 \
    --iter             200 \
    --n_mut            1 \
    --steps            5000000 \
    --temperature      2.0 \
    --out_base         system
```

> **Note:** `main.py` automatically looks for `step0_results.json` inside `<base_dir>/`. If `step_0.py` was run with a custom `--results_path`, copy or symlink the file to `<base_dir>/step0_results.json` before running `main.py`.

#### `main.py` arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--base_dir` | `str` | *(required)* | Root directory for all I/O; `step0_results.json` must exist here |
| `--input_rece_file` | `str` | *(required)* | Receptor PDB filename |
| `--ref_lig_file` | `str` | *(required)* | Reference ligand PDB filename |
| `--ref_seq` | `str` | `CAAAAAAAAAAAC` | Starting peptide sequence |
| `--keep_pos` | `int+` | `[0, 1]` | Residue positions (0-based) held fixed during mutation |
| `--iter` | `int` | `201` | Total number of MC iterations |
| `--n_mut` | `int` | `1` | Number of simultaneous mutations per step |
| `--steps` | `int` | `5 000 000` | MD integration steps per iteration |
| `--step_size` | `float` | `0.002` | MD step size (ps) |
| `--temperature` | `float` | `2.0` | Initial Metropolis temperature $T_0$ |
| `--out_base` | `str` | `system` | Base name for MD output files |

---

## Output

```
<base_dir>/
├── step0_results.json       # Reference docking scores written by step_0.py
├── inputs/
│   ├── ref/pdb/             # Receptor & reference ligand (PDB)
│   ├── ref/pdbqt/           # Receptor & reference ligand (PDBQT)
│   ├── ligands/pdb/         # AF3-predicted peptide structures
│   ├── ligands/pdbqt/       # Converted peptide structures
│   └── ligands/json/        # AF3 input JSON files
├── docking/
│   ├── inputs/              # Post-MD chain-split structures
│   ├── minimized_outputs/   # Vina-minimized poses
│   └── outputs/             # Vina docking poses (PDBQT + PDB)
├── output/
│   ├── complexes/           # MD trajectories and reimaged PDBs
│   └── analysis/            # docking_scores.csv, docking_scores.png
├── checkpoints/             # Pickle checkpoints for fault tolerance
├── global_results/          # Aggregated post-processing results
├── results_top10/           # Top-10 binding sequences
└── log.log                  # Rotating log file
```

### Score file schema

`output/analysis/docking_scores.csv`:

| Column | Description |
|---|---|
| `iteration` | MC iteration index |
| `position` | Mutated residue position(s) |
| `sequence` | Peptide sequence at this iteration |
| `score_pre_simulation` | Vina docking score before MD (kcal/mol) |
| `score_post_simulation` | Vina rescoring score after MD (kcal/mol) |
| `delta_e` | $\Delta E$ used in Metropolis criterion |

---

## Fault Tolerance

ALLIANCE saves a pickle checkpoint every 10 iterations containing the current sequence, best score, temperature, and iteration index. If execution is interrupted, the pipeline automatically resumes from the latest checkpoint.

`step0_results.json` is written only once (by `step_0.py`) and is never overwritten by `main.py`, so the reference baseline remains stable across restarts.

---

## License

Copyright © 2026 FRACTALS Research Group. All rights reserved.  
Visit [https://fractals.group](https://fractals.group) for more information.
