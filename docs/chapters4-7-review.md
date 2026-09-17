# Chapters 4-7: revision notes

> Historical correction record. The later [full-course teaching review](full-course-review.md) supersedes the notebook counts, dependency scope, and runtime totals below.


## Scope

This review covers [Chapter 4](../Chapter04.ipynb), [Chapter 5](../Chapter05.ipynb), [Chapter 6](../Chapter06.ipynb), and [Chapter 7](../Chapter07.ipynb). It continues the [Chapters 1-3 revision](chapters1-3-review.md), preserving the progression from quantum theory and basis functions to wavefunction, density-functional, and semiempirical calculations.

The required examples perform numerical calculations or run real electronic-structure engines. They use embedded molecular inputs and locally installed basis data. Larger exercises are labeled separately. Chapters 8-11 and the legacy shared `utils.py` remain outside this review.

## Main corrections

| Area | Problem addressed | Revised treatment |
| --- | --- | --- |
| Quantum states and observables | A one-particle orbital could be confused with the complete many-electron state; amplitude and probability were insufficiently distinguished | Space-spin coordinates, normalization, probability density, expectation values, and effective one-electron equations are stated separately |
| Molecular Hamiltonian | Coulomb terms mixed unit conventions, with insufficient explanation of pair sums | An explicit atomic-unit Hamiltonian, correct signs, distinct-pair counting, and the corresponding SI Coulomb factor |
| Born-Oppenheimer approximation | Nuclear kinetic energy was described as becoming zero | Fixed-nuclei electronic calculations are separated from nuclear motion on an energy surface; derivative couplings and the nuclear-repulsion convention are explained |
| Basis functions | Slater and Gaussian curves used inconsistent normalization and omitted primitive normalization factors | Normalized 3D Slater functions, normalized s primitives, overlap-based contraction normalization, and independent numerical norm integrals |
| STO-3G/STO-6G comparison | A signed coordinate was labeled radius; the plotted Slater target did not match the supplied H contraction | Labeled amplitude line cuts, spatial density and radial probability plots, separate exact-H and fitted Slater references, and variational trial-energy checks |
| Hartree-Fock and correlation | Imprecise accounts of determinants, exchange, correlation, basis error, and computational scaling | A normalized Slater determinant, distinctions between exchange and correlation, finite-basis versus method error, and qualified descriptions of MP2, CI, and coupled cluster |
| SCF and geometry optimization | Electronic convergence, nuclear convergence, and chemical accuracy could be conflated | Separate electronic and geometry loops, explicit numerical criteria, final-gradient checks, and an independent final-energy recomputation |
| Geometry and scans | Optimization coordinates could be displayed with the wrong units; scan angles and changing global state obscured interpretation | Checked bohr/angstrom conversion, preserved starting coordinates, independent torsion-scan geometries, and relative energies for a specified rigid scan |
| Glucose dependency | Chapter 5 relied on an untracked and stereochemically unverified PDB file | A complete required water optimization; a separate glucose extension using an explicitly specified alpha-D-glucopyranose SMILES and a PubChem source |
| DFT foundations | Density normalization, theorem scope, exchange-correlation terms, and orbital meanings needed correction | Occupation factors, admissible densities, the exact-functional variational principle, kinetic correlation, and local versus generalized Kohn-Sham equations |
| DFT methods and examples | Functional families and software roles were blurred; calculation claims did not consistently match the performed operation | Correct functional classification, separate basis/grid/functional errors, actual water single points and optimization, vibrational checks, and a correctly labeled butadiene single point |
| Semiempirical methods | Method families, implementations, convergence, and energy definitions were insufficiently separated | NDDO, DFTB, and xTB are distinguished; MOPAC runs named models with explicit charge/spin choices and checks its output files and completion messages |
| Thermochemical interpretation | Semiempirical heats of formation could be compared as though they shared the HF/DFT total-energy zero | Defined MOPAC heat-of-formation convention, an explicitly gas-phase NIST water reference, and within-method energy differences for a rigid angle scan |
| Reproducibility and teaching | Stale outputs, hidden dependencies, and limited self-checks | Objectives, exercises, cited assumptions, explicit settings, local inputs, checked outputs, and a shared fresh-kernel validator |

## Required calculations and checks

### Chapter 4: quantum mechanics and basis functions

- Hydrogen STO-3G and STO-6G exponents and coefficients were checked against Basis Set Exchange **version 1** records.
- Primitive and contracted norms are checked by three-dimensional quadrature, independently of the analytic overlap normalization.
- The exact isolated-H 1s reference has exponent 1; the supplied contractions fit a Slater exponent near 1.24. Both references are shown explicitly.
- The exact-H radial maximum and mean radius are checked against 1 and 1.5 bohr, respectively.
- One-electron trial energies are checked against approximately -0.46658185 hartree for STO-3G and -0.47103905 hartree for STO-6G, and against the -0.5 hartree variational lower limit.

### Chapter 5: wavefunction calculations with Psi4

- A prepared 1,3-butadiene conformer supplies an RHF/STO-3G single point and a **13-point rigid torsion scan**. Endpoint periodicity and coordinate preservation are checked.
- Distorted water is optimized at RHF/3-21G. Checks cover electron count, finite energies and gradients, optimization completion, and final gradient size.
- Optimization-history coordinates are converted from bohr to angstroms, compared against the original and final geometries, and exported with valid PDB frame records.
- A new single point at the exported final geometry must reproduce the optimized energy.

### Chapter 6: density functional calculations with Psi4

- A radial integral checks the hydrogen density's electron count.
- PBE and PBE0 water single points use the same geometry and def2-SVP basis. Density-matrix/overlap traces check electron counts; PBE0's exact-exchange fraction is checked explicitly.
- A finer integration grid probes numerical quadrature sensitivity while preserving the functional and geometry.
- A PBE water optimization is followed by an independently evaluated gradient and a vibrational calculation. Three positive internal modes support a local minimum at this computational level.
- A B3LYP/def2-SVP single point is evaluated on UFF-prepared butadiene, with molecular formula, coordinates, and electron count checked. It is explicitly not described as a DFT geometry optimization.
- Settings, geometries, numerical tables, and a calculation record are saved under `outputs/chapter06/`.

### Chapter 7: semiempirical calculations with MOPAC

- MNDO, AM1, PM3, PM6, and PM7 single points use the same neutral water geometry; each method also optimizes the same distorted starting structure.
- A fresh directory is created for each job. Checks cover executable status, required files, normal termination, SCF success, requested method, array lengths, finite values, electron count, and charge sum.
- Single-point coordinates must remain unchanged. Optimizations must meet the specified nuclear gradient criterion.
- Optimized model heats of formation are compared with the cited gas-phase water reference, with phase and unit conversion stated.
- A **nine-point PM7 rigid angle scan** uses fixed O-H distances and relative energies within one method. This is distinct from a relaxed path or thermal distribution.

## Validation record

On 2026-09-15, all **54 code cells** in Chapters 4–7 passed execution in separate fresh kernels. The final run used `--inplace`, and these four source notebooks contain the verified outputs. Counts and timings below match the recorded validator results.

| Notebook | Code cells in current source | Final fresh-kernel result | Elapsed time |
| --- | ---: | --- | --- |
| [Chapter 4](../Chapter04.ipynb) | 13 | Passed | 2.61 s |
| [Chapter 5](../Chapter05.ipynb) | 21 | Passed | 11.84 s |
| [Chapter 6](../Chapter06.ipynb) | 11 | Passed | 57.61 s |
| [Chapter 7](../Chapter07.ipynb) | 9 | Passed | 2.41 s |
| **Total** | **54** | **Passed** | **74.47 s** |

From the activated course environment, the final command was:

```sh
python scripts/validate_chapters.py Chapter04.ipynb Chapter05.ipynb Chapter06.ipynb Chapter07.ipynb --inplace
```

The [validator](../scripts/validate_chapters.py) uses a separate fresh kernel for each notebook, rejects execution errors, validates notebook structure, and writes the actual environment and results to `outputs/validation/report.json`. It updates a source notebook only after that notebook executes successfully. Reports are generated locally and may be replaced by a later validation run; they are not tracked source data.

The four chapters' plots and molecular depictions were visually inspected during development, and the scientific content received peer review. Final execution verifies numerical reproducibility in the tested environment; it does not establish experimental accuracy or validate every version permitted by dependency ranges.

### Chapters 1–3 compatibility check

The six current Chapters 1–3 notebooks also passed in the expanded environment below, executing **187 code cells**. This was a separate invocation without `--inplace`; its executed notebook copies are under `outputs/validation/`. Those source notebooks currently have cleared outputs. The earlier [185-cell validation record](chapters1-3-review.md#validation) is preserved as a historical record of that earlier revision; current source counts have increased.

| Notebook | Executed code cells | Result | Elapsed time |
| --- | ---: | --- | --- |
| [Chapter 1, Part 1](../Chapter01_Part1.ipynb) | 45 | Passed | 1.22 s |
| [Chapter 1, Part 2](../Chapter01_Part2.ipynb) | 29 | Passed | 2.78 s |
| [Chapter 1, Part 3](../Chapter01_Part3.ipynb) | 57 | Passed | 2.05 s |
| [Chapter 2, Part 1](../Chapter02_Part1.ipynb) | 17 | Passed | 1.72 s |
| [Chapter 2, Part 2](../Chapter02_Part2.ipynb) | 21 | Passed | 2.38 s |
| [Chapter 3](../Chapter03.ipynb) | 18 | Passed | 31.38 s |

Together, the two invocations passed **241 code cells across ten notebooks**. Their results were combined into the local validation report, retaining each notebook's observed time.

### Tested environment

Native Windows, conda-forge environment:

| Component | Version |
| --- | --- |
| Python | 3.12.14 |
| Psi4 | 1.11 |
| MOPAC | 23.2.5 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| RDKit | 2026.3.6 |
| Matplotlib | 3.11.2 |

See the [setup instructions](../Readme.md#set-up-python), [Conda environment](../environment.yml), and [Python requirements](../requirements-chapters1-7.txt). Psi4 and MOPAC are installed through Conda, not by the pip requirements alone. On Windows, activate the environment before launching Jupyter or validation so its native libraries, including those in `Library/bin`, are on `PATH`.

## Interpretation and remaining limitations

- Core computations run offline after installation. The optional py3Dmol branch in Chapter 5 and its browser controls are outside the required execution path; they may load JavaScript from the internet. The extended glucose optimization is not part of the required run.
- Chapter 4 studies s functions on one center. It does not implement general angular-momentum integrals or a molecular SCF program. STO-6G's improved fit in the stated metric does not establish a universal accuracy ordering of basis sets.
- The small HF and DFT calculations use specified finite bases and numerical settings. They are teaching examples, not basis-converged barrier heights, thermochemical benchmarks, or experimental predictions.
- SCF convergence, nuclear stationarity, local-minimum character, and model accuracy answer different questions. Chapter 6 checks water's vibrational curvature; the Chapter 5 HF and Chapter 7 semiempirical optimizations do not include a Hessian classification. None establishes a global minimum.
- Optimization histories have no physical timestep. Rigid scans are not relaxed potential-energy paths, transition-state searches, or equilibrium distributions.
- Approximate DFT total energies do not provide a general variational ranking across functionals. An orbital eigenvalue gap is not automatically an optical excitation energy.
- MOPAC's parameterized heats of formation use different reference conventions from the HF/DFT molecular energies. Unit conversion alone cannot make these quantities interchangeable. Partial atomic charges are model-dependent partitions.
- One gas-phase water reference does not validate semiempirical transferability. Water may overlap with model parameterization data, and five predictions for one molecule are not five independent experimental observations.
- Solvent, temperature, zero-point energy, entropy, and standard-state choices require explicit treatment for the relevant observable. The MOPAC heat-of-formation convention also means that an ab initio thermal-correction recipe should not be added automatically.
- The pre-existing `structures/glucose.pdb` remains an unverified legacy file. The revised required calculations and glucose extension do not rely on it. Subsequent reviews cover [Chapter 8](chapter8-review.md), [Chapter 9](chapter9-review.md), and [Chapters 10–11](chapters10-11-review.md). The legacy `utils.py` still requires its own maintenance review.

## Sources and local navigation

The notebooks place references beside the relevant equations and examples. Principal sources include:

- [Basis Set Exchange STO-3G data](https://www.basissetexchange.org/api/basis/sto-3g/format/json/?elements=1&version=1), [STO-6G data](https://www.basissetexchange.org/api/basis/sto-6g/format/json/?elements=1&version=1), and [GBasis normalization documentation](https://gbasis.qcdevs.org/_autosummary/gbasis.html).
- [Born and Oppenheimer's original paper](https://doi.org/10.1002/andp.19273892002) and [NIST atomic-unit conventions](https://www.nist.gov/publications/units-and-constants).
- [Psi4 HF theory](https://psicode.org/psi4manual/master/scf.html), [optimization API](https://psicode.org/psi4manual/master/api/psi4.driver.optimize.html), and [DFT implementation](https://psi4.github.io/psi4docs/master/dft.html).
- [Hohenberg and Kohn](https://doi.org/10.1103/PhysRev.136.B864), [Kohn and Sham](https://doi.org/10.1103/PhysRev.140.A1133), and [Levy's constrained search](https://pmc.ncbi.nlm.nih.gov/articles/PMC411802/).
- [MOPAC input guide](https://openmopac.net/guides/basics/), [heat-of-formation definition](https://openmopac.net/Manual/SCF_calc_hof.html), and [NIST gas-phase water thermochemistry](https://webbook.nist.gov/cgi/cbook.cgi?Name=water&cTC=on&cTG=on).
- [PubChem alpha-D-glucose](https://pubchem.ncbi.nlm.nih.gov/compound/Alpha-D-Glucose) for the Chapter 5 extended exercise's specified stereoisomer.

Course notebooks: [Chapter 4](../Chapter04.ipynb), [Chapter 5](../Chapter05.ipynb), [Chapter 6](../Chapter06.ipynb), and [Chapter 7](../Chapter07.ipynb). The [earlier revision notes](chapters1-3-review.md) retain the actual validation record from the first review.
