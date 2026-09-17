# Teaching review: Chapters 4–7

## Scope and approach

All cells of the four existing notebooks were reviewed. The revision keeps their checked quantum-chemistry coverage and adds a first-reading route, physical intuition before equations, definitions where new symbols appear, and small research decisions supported by actual calculations. Dense method taxonomies and implementation helpers are marked as deeper reading. The numerical examples run offline in the course environment; no experimental-looking data or performance scores were invented.

Notebook Markdown uses only `$...$` and `$$...$$` for mathematics. Source outputs were cleared for the final course-wide execution. Author validation retains executed copies and a separate report under `outputs/full_review_4_7/validation/`.

## Chapter 4: from states to a basis decision

- Introduced the difference between a state, an orbital, and a basis function before the more formal machinery. Added explicit meanings for operators, eigenvalues, expectation values, and the generalized orbital eigenproblem.
- Added a normalized particle-in-a-box illustration: amplitudes, stationary probabilities, and coherent superpositions at three relative phases. The plot is explicitly an abstract one-dimensional model, not a molecular spectrum.
- Reframed the molecular Hamiltonian as an energy inventory and connected the Born–Oppenheimer approximation to the electronic/nuclear separation used in molecular dynamics.
- Added an actual hydrogen variational calculation comparing a fixed STO-nG contraction with optimized coefficients in its independent primitive-function space. The Hamiltonian, exponents, and electron count stay fixed within each comparison.
- The calculated energies are −0.46658185 to −0.49574080 hartree for the three-function space and −0.47103905 to −0.49985967 hartree for the six-function space. The exact hydrogen ground-state energy is −0.5 hartree. This is a basis-flexibility check, not an electron-correlation calculation or a universal ranking of published basis sets.

New figures show states/probabilities and the energy cost of fixing contraction coefficients. Existing normalization, Gaussian/Slater comparisons, and basis-record provenance remain.

## Chapter 5: separate the geometry, electronic iteration, and model

- Added a two-electron antisymmetrized example before the Slater determinant, and a plain-language distinction between exchange and beyond-HF correlation.
- Added an SCF/geometry diagram showing the electronic inner loop and nuclear outer loop. Defined the residual and the role of DIIS.
- Restored the butadiene scan to 13 actual HF points, consistent with its teaching purpose and short-cell budget; the current pre-review source had 51 executable points. Added three calculated-geometry snapshots and an explicitly coarse sampled curve.
- Added a fixed-geometry water experiment separating a basis change at HF from frozen-core MP2 correlation at a fixed basis. Calculations are HF/STO-3G, HF/3-21G, and frozen-core MP2/3-21G, all at the same HF/3-21G optimized geometry. Their energies are approximately −74.962585, −75.585960, and −75.706952 hartree.
- Checked that the reported MP2 correlation contribution matches MP2 minus HF at the same geometry and basis. Clarified that MP2 is not variational and the diagram's energy changes are not errors against an exact reference.
- Moved the optional py3Dmol viewer into a Markdown snippet; Run All uses self-contained static figures. Retained the actual optimization history, units, gradient checks, and export helpers.

## Chapter 6: connect density to a usable property comparison

- Introduced functions versus functionals and basis versus integration grid before the formal DFT results. Added a physical energy inventory for the Kohn–Sham terms and symbol definitions before the orbital equation.
- Added actual PBE and PBE0 molecular-plane density slices with a common logarithmic scale, plus a signed density-difference map. Both spin densities are included. Explained why a slice integral is not an electron count; the three-dimensional basis-space trace is checked separately.
- Added a PBE0/def2-TZVP single point and a controlled dipole comparison that changes grid, orbital basis, or functional one at a time at one fixed water geometry. The illustrative 0.001 D resolution is a reporting target, not an experimental uncertainty or certified model error.
- Reduced the repeated-gradient baseline grid from 99 radial/590 angular points to 50/194 with robust pruning. Retained the actual 150/974 grid check: its PBE0 energy change is about $1.97\times10^{-6}$ hartree and its dipole change about $4\times10^{-6}$ D. The basis change is about 0.0503 D; the PBE-versus-PBE0 difference is about 0.0802 D. These sensitivities are not additive or a complete uncertainty estimate.
- Kept the actual PBE geometry optimization and full finite-difference vibrational check; the three positive internal modes are approximately 1609, 3691, and 3790 inverse centimetres. Added a geometry/curvature figure and distinguished energy lowering, stationarity, and minimum classification.
- Kept the B3LYP butadiene single point, with physical-model limitations explicit. Longer extension work is framed as a study-design exercise rather than another Run All calculation.

The smaller grid is a measured teaching compromise for these examples, not a recommendation for production calculations. It reduced the frequency cell from approximately 26 seconds to below 9 seconds in the tested environment. Grid, basis, functional, geometry, and environment errors remain distinct.

## Chapter 7: test a practical preoptimization handoff

- Introduced fitted parameters, valence/core treatment, transferability, and method selection through concrete research decisions before NDDO notation.
- Explained water's distinct atom, electron, and basis-function counts: three atoms, ten all-electron electrons, eight explicit valence electrons, and six valence basis functions in this MOPAC example.
- Moved formation-enthalpy interpretation ahead of the numerical tables. Added the balanced gas-phase formation reaction and emphasized that changing units does not align MOPAC formation enthalpies with HF clamped-nuclei energy definitions.
- Added a fixed-geometry comparison of oxygen partial charges and molecular dipoles. Clarified atomic partition dependence and MOPAC's intra-atomic dipole contributions.
- Added two actual HF/STO-3G gradient calculations on the distorted water input and PM7-prepared geometry. The latter lowers the HF energy by about 108.05 kJ/mol and decreases its Cartesian gradient RMS from about 0.085 to 0.030 hartree/bohr. This supports a starting-geometry decision for this deliberately strained example; it does not establish an HF stationary point or general method accuracy.
- Kept all five MOPAC methods, convergence/output checks, the NIST formation-enthalpy comparison, and the bounded rigid-angle scan. The subprocess timeout is now 10 seconds. Added explanatory answers connecting the new calculations to the limitations.

## Validation and visual inspection

Fresh-kernel validation on Windows, Python 3.12.14, Psi4 1.11, MOPAC 23.2.5, RDKit 2026.03.6, NumPy 2.5.3, and Matplotlib 3.11.2 passed with a 10-second cell timeout. The course environment uses one numerical thread and `MKL_THREADING_LAYER=SEQUENTIAL`.

| Notebook | Code cells | Fresh-kernel notebook elapsed time | Slowest cell |
|---|---:|---:|---:|
| Chapter 4 | 15 | 3.34 s | 0.93 s |
| Chapter 5 | 23 | 8.69 s | 2.96 s |
| Chapter 6 | 14 | 21.83 s | 8.40 s |
| Chapter 7 | 11 | 4.20 s | 1.08 s |

All 63 code cells passed. Notebook times include kernel startup, execution, rendering, and saving; individual cell times include imports and rendering where applicable. They are machine-dependent measurements, not universal limits. Earlier direct executions also passed. All new figures were rendered and visually inspected for legible titles, units, scales, and complete labels. Assertions cover normalization, variational bounds within the stated spaces, SCF results, charge/electron counts, geometry preservation, energy/gradient consistency, and positive internal vibrational curvature.

## Sources checked for new or expanded material

- [MIT physical chemistry: particle in a box](https://www.ocw.mit.edu/courses/5-61-physical-chemistry-fall-2007/187f992fd0cde12595f74686872a4dd5_lecture8.pdf): normalized states and their probability interpretation.
- [SciPy generalized Hermitian eigensolver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html): generalized eigenvectors and overlap normalization.
- [Psi4 MP2 documentation](https://psicode.org/psi4manual/master/dfmp2.html): MP2 implementation and frozen-core treatment; [Psi4 method capabilities](https://psicode.org/psi4manual/1.9.x/capabilities.html): conventional MP2 option.
- [Psi4 BasisSet API](https://psicode.org/psi4manual/master/api/psi4.core.BasisSet.html#psi4.core.BasisSet.compute_phi) and [official AO evaluation sample](https://github.com/psi4/psi4/blob/master/samples/phi-ao/input.dat): evaluating density slices from basis functions.
- [Psi4 1.11 frequency documentation](https://psicode.org/psi4manual/1.11.x/freq.html): finite differences of gradients and vibrational analysis.
- [MOPAC basic usage](https://openmopac.net/guides/basics/), [formation-enthalpy components](https://openmopac.net/Manual/SCF_calc_hof.html), and [thermochemical conventions](https://openmopac.net/Manual/thermochemistry.html): energy definitions and parameterization targets.
- [MOPAC property definitions](https://openmopac.net/Manual/features.html): atomic charges and molecular dipoles; [geometry optimization](https://openmopac.net/Manual/geometry_optimizer.html): force-based relaxation.

Original primary references for Born–Oppenheimer, Gaussian basis records, HF, Hohenberg–Kohn, Kohn–Sham, functional families, PM7, and NIST thermochemistry remain linked in the notebooks.
