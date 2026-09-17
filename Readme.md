# Cheminformatics Course

Jupyter notebooks for the **Cheminformatics** course at [Ho Chi Minh City University of Technology and Education (HCMUTE)](https://en.hcmute.edu.vn/).

## Chapters 1–12: graduate teaching edition

Start with the [course guide](docs/course-guide.md): it provides a learning route, short chemistry/math refreshers, research projects, and an assessment rubric. The course assumes introductory chemistry and introduces Python from the beginning. Each chapter now includes a beginner bridge, purposeful visual explanations, and worked applications with explicit limits. First-pass material and deeper derivations are identified so students can build understanding gradually.

| Notebook | Main topics |
| --- | --- |
| [Chapter 1, Part 1](Chapter01_Part1.ipynb) | Python, Jupyter, and reproducible execution |
| [Chapter 1, Part 2](Chapter01_Part2.ipynb) | NumPy, pandas, tables, and plotting |
| [Chapter 1, Part 3](Chapter01_Part3.ipynb) | RDKit, SMILES, stereochemistry, and molecular coordinates |
| [Chapter 2, Part 1](Chapter02_Part1.ipynb) | Bond lengths, bond angles, and signed dihedrals |
| [Chapter 2, Part 2](Chapter02_Part2.ipynb) | Force fields, conformational energy, and geometry optimization |
| [Chapter 3](Chapter03.ipynb) | Dynamics, ensembles, integration, and OpenMM simulations |
| [Chapter 4](Chapter04.ipynb) | Quantum states, the molecular Hamiltonian, and normalized basis functions |
| [Chapter 5](Chapter05.ipynb) | Hartree-Fock, correlation, SCF, torsion scans, and Psi4 optimization |
| [Chapter 6](Chapter06.ipynb) | Density functional theory and practical Psi4 calculations |
| [Chapter 7](Chapter07.ipynb) | Semiempirical methods, MOPAC, and interpreting heats of formation |
| [Chapter 8, Part 1](Chapter08_Part1.ipynb) | Neutral molecules, dipoles, density, electrostatic potential, and solvent models |
| [Chapter 8, Part 2](Chapter08_Part2.ipynb) | Ions, radicals, spin, and balanced energy comparisons |
| [Chapter 8, Part 3](Chapter08_Part3.ipynb) | Molecular orbitals, occupations, and orbital gaps |
| [Chapter 8, Part 4](Chapter08_Part4.ipynb) | Controlled potential-energy scans and surface interpretation |
| [Chapter 8, Part 5](Chapter08_Part5.ipynb) | Vibrations, IR intensities, and thermochemistry |
| [Chapter 8, Part 6](Chapter08_Part6.ipynb) | Electronic excitations and absorption spectra |
| [Chapter 8, Part 7](Chapter08_Part7.ipynb) | Quantum forces and short Born-Oppenheimer dynamics |
| [Chapter 9, Part 1](Chapter09_Part1.ipynb) | Reaction thermodynamics, activities, equilibrium constants, and standard states |
| [Chapter 9, Part 2](Chapter09_Part2.ipynb) | Transition-state verification, short IRC segments, and endpoint refinement |
| [Chapter 9, Part 3](Chapter09_Part3.ipynb) | Eyring rates, reversible kinetics, catalysis, and product selectivity |
| [Chapter 9, Part 4](Chapter09_Part4.ipynb) | Reaction SMILES/SMARTS, atom mapping, balance, and stereochemical data checks |
| [Chapter 10, Part 1](Chapter10_Part1.ipynb) | Molecular descriptors, fingerprint generators, similarity, and information loss |
| [Chapter 10, Part 2](Chapter10_Part2.ipynb) | Dataset audits, scaffold splits, regression/classification, baselines, and evaluation |
| [Chapter 11, Part 1](Chapter11_Part1.ipynb) | Perceptrons, differentiable units, gradients, PyTorch, and linear baselines |
| [Chapter 11, Part 2](Chapter11_Part2.ipynb) | Multilayer perceptrons, validation stopping, tensor shapes, and complete checkpoints |
| [Chapter 11, Part 3](Chapter11_Part3.ipynb) | Molecular property learning, label audits, and reaction-yield representations and splits |
| [Chapter 11, Part 4](Chapter11_Part4.ipynb) | Graph neural networks, message passing, permutation tests, and stereochemical limitations |
| [Chapter 12, Part 1](Chapter12_Part1.ipynb) | Molecular graph features, directed edges, stereochemistry, and batching |
| [Chapter 12, Part 2](Chapter12_Part2.ipynb) | Bond-aware message passing, aggregation, readout, and GNN architectures |
| [Chapter 12, Part 3](Chapter12_Part3.ipynb) | Measured solubility prediction, scaffold splits, baselines, and model checkpoints |
| [Chapter 12, Part 4](Chapter12_Part4.ipynb) | Graph-model limitations, diagnostics, and attribution checks |
| [Chapter 12, Part 5](Chapter12_Part5.ipynb) | 3D geometric networks, symmetry, differentiable energy, and equivariant forces |
| [Chapter 12, Part 6](Chapter12_Part6.ipynb) | Real PyTorch Geometric: Data, Batch, DataLoader, MessagePassing, GCNConv, and GINEConv |
| [Chapter 12, Part 7](Chapter12_Part7.ipynb) | PyG solubility regression, mini-batches, validation checkpoints, and matched baselines |
| [Chapter 12, Part 8](Chapter12_Part8.ipynb) | PyG BBBP classification, screening thresholds, and GNNExplainer audits |

All 35 notebooks have been reviewed for graduate students entering the field. The [full-course review](docs/full-course-review.md) records changes, final execution checks, and scientific limitations. Detailed teaching-review notes cover [Chapters 1–3](docs/full-review-chapters1-3.md), [4–7](docs/full-review-chapters4-7.md), [8–9](docs/full-review-chapters8-9.md), [10](docs/full-review-chapter10.md), [11](docs/full-review-chapter11.md), and [12](docs/full-review-chapter12.md). The [dataset notes](datasets/README.md) document the bundled measurements and label definitions.

### Chapter 12: learning from molecular graphs

Chapter 11 Part 4 is a short introduction. Chapter 12 now contains **eight standalone notebooks**. Parts 1–5 implement graph operations directly in PyTorch so students can inspect the mathematics. Parts 6–8 use actual PyTorch Geometric classes for batching, graph layers, measured-property regression, classification, and explanations. The [Chapter 12 learning route](docs/course-guide.md#chapter-12-foundations-and-real-pytorch-geometric) connects the two implementations.

The measured-data experiments compare models on identical rows with explicit train/validation/test separation. Small baselines can outperform GNNs here; students interpret that evidence rather than assume an architecture is always best. Diagnostic examples separate mathematical behavior from chemical accuracy. The 3D lesson connects conformers, energies, forces and symmetry to earlier chapters. No GPU, pretrained-model download, or compiled graph extension is required.

## Set up Python

Use **64-bit Python 3.12** and a separate environment for the course. Run the commands below in a terminal from this repository's root folder. Installation requires internet access; a GPU is not required for these introductory exercises.

### Option A: Conda / Anaconda Prompt for the complete course

```sh
conda env create -f environment.yml
conda activate cheminformatics
python -m jupyterlab
```

If an older `cheminformatics` environment already exists, use `conda env update -f environment.yml` instead of the create command. Deactivate and reactivate that environment after updating so its environment variables are refreshed, then launch JupyterLab.

On Windows, activation also puts the environment's native libraries on `PATH`. Launch both JupyterLab and validation from that activated terminal.

The [environment file](environment.yml) installs Python 3.12, **Psi4 1.11**, and **MOPAC 23.2.5** from conda-forge, then the Python packages in [requirements-chapters1-12.txt](requirements-chapters1-12.txt). Chapters 10–11 add **scikit-learn 1.8** and **PyTorch 2.11.0**; Chapter 12 Parts 6–8 add **PyTorch Geometric 2.8.0.post1**, the release installed and tested for these lessons. Windows/Linux installations use CPU wheels from PyTorch's official package index; macOS uses its native PyPI wheel. No GPU or TensorBoard is required. The PyG lessons use basic installation features without optional compiled graph packages; see the [official installation guide](https://pytorch-geometric.readthedocs.io/en/2.8.0/install/installation.html). Installing the pip requirements alone does not install the quantum-chemistry engines.

The environment also sets `MKL_THREADING_LAYER=SEQUENTIAL` before Python starts. This selects [MKL's supported sequential backend](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2024-0/dynamic-select-the-interface-and-threading-layer.html), avoiding the Windows OpenMP conflict found when combining the tested Conda numerical stack with PyTorch. The short examples use one CPU thread. See [PyTorch installation guidance](https://pytorch.org/get-started/locally/) for platform-specific wheel support.

Chapter 4's numerical examples need no quantum-chemistry engine. Chapters 5–7, several parts of Chapter 8, and Chapter 9 Part 2 use Psi4 for actual electronic-structure calculations; Chapter 7 runs the MOPAC executable. Select the kernel belonging to this environment so the notebooks can find the required software. See [Psi4 installation guidance](https://psicode.org/installs/latest/) and [MOPAC documentation](https://openmopac.net/Manual/) for engine details.

### Option B: Python virtual environment for Chapters 1-3 only

This route installs the software for the first six notebooks. Use the Conda environment above when continuing to the full Chapters 1-12 sequence.

On Windows, use these commands in PowerShell (activation is unnecessary):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-chapters1-3.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

On macOS or Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-chapters1-3.txt
.venv/bin/python -m jupyterlab
```

These instructions use the [RDKit installation](https://www.rdkit.org/docs/Install.html) and [OpenMM installation](https://docs.openmm.org/latest/userguide/application/01_getting_started.html) routes. The package name on PyPI is `rdkit`; `rdkit-pypi` is its older name.

## Work through a notebook

1. Open the notebook in JupyterLab and select the Python kernel from your course environment.
2. Run cells from top to bottom. Before saving completed work, use **Restart Kernel and Run All Cells** to check that no variables depend on an earlier, out-of-order run.
3. Read the units and model assumptions alongside each result. Electronic SCF convergence, geometry convergence, and chemical accuracy are separate checks. Short teaching trajectories do not establish equilibrium or biological behavior.
4. Try the exercises before opening the suggested answers, where provided.

Core calculations use embedded examples, tracked course data, or data distributed with the installed packages. They run offline after setup. Optional interactive 3D viewers may need internet access in the browser to load 3Dmol.js; static depictions and plots provide offline alternatives. Larger calculations are explicitly labeled as extended exercises.

Generated structures, trajectories, calculation logs, and validation results belong in `outputs/`, which Git ignores. Keep the original molecular inputs and method/basis/unit information when interpreting or comparing results.

## Check the revised notebooks

From the activated Conda environment, execute all **35 notebooks** in fresh kernels with the short-cell check:

```sh
python scripts/validate_chapters.py --cell-timeout 10
```

The validator uses the current interpreter, raises cell errors, and saves executed copies plus package versions and per-cell timings in `outputs/validation/`. Add `--inplace` to save successful runs into the source notebooks. The [full-course review](docs/full-course-review.md#final-validation) records the final measured runtimes. Timings depend on the computer and software build; a timeout diagnoses a runtime problem rather than certifying scientific accuracy.

For selected chapters or notebooks:

```sh
python scripts/validate_chapters.py --chapter8 --cell-timeout 10
python scripts/validate_chapters.py --chapter9 --cell-timeout 10
python scripts/validate_chapters.py --chapters10-11 --cell-timeout 10
python scripts/validate_chapters.py --chapter12 --cell-timeout 10
python scripts/validate_chapters.py Chapter05.ipynb Chapter06.ipynb Chapter07.ipynb --cell-timeout 10
```

Selectors `--chapter10` and `--chapter11` are also available. The default without a timeout argument is 300 seconds per cell, useful for diagnosis on slower machines; the published teaching review uses the explicit 10-second guard. Calculation sizes, training epochs, scans and trajectories are bounded, and no notebook's default run downloads data or a pretrained model.

For the Chapters 1–3 virtual environment, select only its six notebooks. On Windows:

```powershell
.\.venv\Scripts\python.exe scripts/validate_chapters.py Chapter01_Part1.ipynb Chapter01_Part2.ipynb Chapter01_Part3.ipynb Chapter02_Part1.ipynb Chapter02_Part2.ipynb Chapter03.ipynb --cell-timeout 10
```

On macOS/Linux, use `.venv/bin/python` with the same arguments. The complete course uses the Conda environment because pip requirements alone do not install Psi4 or MOPAC.

Engine versions are specified in `environment.yml`; several Python dependencies allow version ranges, so this is not a complete lockfile. Consult the generated version report when reproducing a run. Earlier scientific-correction records are retained as historical documents: [1–3](docs/chapters1-3-review.md), [4–7](docs/chapters4-7-review.md), [8](docs/chapter8-review.md), [9](docs/chapter9-review.md), [10–11](docs/chapters10-11-review.md), [initial Chapter 12](docs/chapter12-review.md). Their earlier cell counts and timings do not describe this full-course edition.

## Contact

Questions and suggestions: [datnp@hcmute.edu.vn](mailto:datnp@hcmute.edu.vn).
