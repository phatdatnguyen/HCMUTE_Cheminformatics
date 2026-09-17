# Chapters 1–3: revision notes

> Historical correction record. The later [full-course teaching review](full-course-review.md) supersedes the notebook counts, dependency scope, and runtime totals below.


## Scope

This review covers the six notebooks from `Chapter01_Part1.ipynb` through `Chapter03.ipynb`. It addresses scientific accuracy, executable examples, and teaching clarity. It preserves the progression from Python and molecular representations to mechanics and dynamics. Subsequent revision notes cover [Chapters 4–7](chapters4-7-review.md), [Chapter 8](chapter8-review.md), [Chapter 9](chapter9-review.md), and [Chapters 10–11](chapters10-11-review.md). The legacy shared `utils.py` remains outside these reviews.

## Main corrections

| Area | Problem in the earlier edition | Revised treatment |
| --- | --- | --- |
| Installation and execution | Obsolete `rdkit-pypi` command, package installation during Run All, old kernel metadata, and implicit environment assumptions | Shared Python 3.12 setup; installation commands in explanatory Markdown; a fresh-kernel validation script |
| Python and scientific tables | Misleading collection/indexing descriptions; array constructor confused with array type; missing view/copy and standard-deviation conventions | Correct collection semantics, `ndarray` versus `np.array`, worked view/copy examples, explicit `ddof`, missing-data handling, and unit-aware chemistry exercises |
| Chemical identity | A different compound labeled Taxol/paclitaxel, plus incorrect small-molecule examples | Verified structures and formula checks, with a PubChem reference for paclitaxel |
| SMILES and stereochemistry | Bracket, aromaticity, and `@`/`@@` explanations could imply incorrect chemical meanings | Explicit parsing checks; local SMILES ordering distinguished from absolute stereochemical labels |
| Coordinates and display | A nominal 3D viewer used unsuitable coordinates/format; XYZ atom order and connectivity were assumed | Molecular graph distinguished from coordinates; explicit coordinate generation and mapping; correct viewer format identifiers |
| Molecular geometry | Incorrect C–C arithmetic and a malformed signed-dihedral equation | Correct distance calculation (about 1.531612 Å for the supplied ethane), signed `atan2`, defined atom order, degeneracy handling, and NumPy/RDKit cross-checks |
| Force-field theory | Incorrect sign for the energy gradient, double-counted nonbonded pairs, and minimization described as finding the most stable conformation | Forces distinguished from gradients; pair counting and model-specific terms explained; local minima distinguished from global minima and equilibrium |
| Conformational calculations | Ethane scan changed global state; glucose work depended on untracked PDB data | Independent conformer copies, controlled scans, parameter/convergence checks, and an embedded stereochemically specified glucose example |
| Dynamics | A symplectic-Euler update labeled forward Euler; unjustified ethane parameters and boundary conditions; confusing time, energy, and ensemble interpretation | Correct integrator comparison against an analytical oscillator, explicit model assumptions and units, and appropriately qualified OpenMM demonstrations |
| Reproducibility and teaching | Missing data dependencies, unseeded calculations, stale outputs, and few learning checks | Self-contained core examples, fixed seeds where supported, objectives, exercises, numerical checks, and regenerated outputs |

The notebooks cite sources next to the relevant explanations and in reference sections. Core references include the [RDKit documentation](https://www.rdkit.org/docs/), [OpenSMILES specification](https://opensmiles.org/opensmiles.html), [PubChem paclitaxel record](https://pubchem.ncbi.nlm.nih.gov/compound/36314), [NumPy documentation](https://numpy.org/doc/stable/), and [OpenMM theory guide](https://docs.openmm.org/latest/userguide/theory.html).

## Validation

On 2026-09-15, all **185 code cells across six notebooks** passed top-to-bottom execution in separate fresh kernels using `python scripts/validate_chapters.py --inplace`. That run saved regenerated outputs to the source notebooks. This is the historical validation record from the first revision. The current Chapters 1–3 sources have cleared outputs; the subsequent [compatibility check](chapters4-7-review.md#chapters-13-compatibility-check) passed 187 code cells and saved executed copies under `outputs/validation/`.

| Notebook | Executed code cells | Result |
| --- | ---: | --- |
| Chapter 1, Part 1 | 45 | Passed |
| Chapter 1, Part 2 | 29 | Passed |
| Chapter 1, Part 3 | 56 | Passed |
| Chapter 2, Part 1 | 17 | Passed |
| Chapter 2, Part 2 | 21 | Passed |
| Chapter 3 | 17 | Passed |

Validation included:

- Chemical identity, stereochemistry, and molecular-format checks in Chapter 1; in-memory CSV/Excel round trips and numerical exercises.
- NumPy/RDKit agreement for geometry, signed-angle and rigid-motion checks, a periodic ethane energy scan, minimization status checks, and an analytic gradient checked against finite differences.
- Numerical integration checked against an analytical harmonic oscillator; ethane force-term counts, finite energies, bond stability, and a 101-frame PDB round trip.
- The bundled solvated-protein example's topology, parameter coverage, periodic box, finite energies, duration, and aligned RMSD. Alignment was also checked against a known rigid transformation. Its startup temperature transient is explicitly discussed as evidence that the short trajectory is not equilibrated.
- Notebook schema, Python syntax, local navigation links, and representative rendered molecular depictions and plots. Clipped 3D axis labels were corrected.

**Tested stack:** Windows, 64-bit Python 3.12.6, with these installed package versions:

| Package | Version |
| --- | --- |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| Matplotlib | 3.10.8 |
| Seaborn | 0.13.2 |
| openpyxl | 3.1.5 |
| RDKit | 2025.9.3 |
| OpenMM | 8.4.0.post2 |
| py3Dmol | 2.5.5 |
| JupyterLab | 4.5.3 |
| ipykernel | 7.1.0 |
| nbformat | 5.10.4 |
| nbclient | 0.10.4 |

The validation environment reused existing system packages and installed missing packages into a workspace virtual environment. A clean Conda installation and every version allowed by the dependency ranges were not tested. The validator records the actual installed versions in `outputs/validation/report.json` on each run.

Optional py3Dmol display branches and Markdown extension snippets were not included in the execution pass. The offline animation was generated successfully; browser interaction with animation controls was not tested. The protein calculation used OpenMM's CPU platform with one thread; the Reference fallback was not part of the final full-notebook run.

## Interpretation and remaining scope

- Static figures and core calculations should be usable without network downloads after installation. Optional py3Dmol displays load JavaScript in the browser and may require internet access.
- An optimization trace is a numerical search history, not elapsed physical time. A short MD trajectory is a classroom demonstration and cannot establish convergence, folding, binding, or experimental agreement.
- Force-field energies depend on the model and its conventions. They are not directly interchangeable with electronic energies, free energies, or energies from another force field.
- Files excluded from Git are not reliable course dependencies. The revised core examples avoid relying on the pre-existing local PDB files.
- The local `structures/glucose.pdb` should not be treated as a verified glucose reference: its assigned stereochemistry did not match the cited alpha-D-glucose structure. The subsequent [Chapters 4–7 review](chapters4-7-review.md) removes Chapter 5's dependency on that file and provides an explicitly specified glucose extension.
- The environment and validation record above document the original Chapters 1–3 pass. The [Chapters 4–7 review](chapters4-7-review.md) records the later quantum-chemistry environment and the current compatibility run; see also the [current setup instructions](../Readme.md#set-up-python).
