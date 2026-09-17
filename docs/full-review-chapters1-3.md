# Beginner-focused review: Chapters 1–3

Reviewed on 2026-09-16 for graduate students beginning cheminformatics. All six notebooks were read in full. Existing correct scientific coverage and reproducible examples were retained, with clearer entry points, definitions, controlled research questions, and interpretation beside the calculations. This report supplements the historical [Chapters 1–3 correction record](chapters1-3-review.md).

## Changes by notebook

| Notebook | Pedagogical changes | Added visual/application work |
| --- | --- | --- |
| [Chapter 1 Part 1](../Chapter01_Part1.ipynb) | Explains package, environment, kernel, assignment, and notebook state without assuming prior programming. Gives a core reading path and separates setup administration from calculations. Defines amount, molar mass, and final solution concentration, with a unit table. Links the unified setup instructions. | A calculated NaCl dilution plan connects dictionaries, loops, conditions, functions, unit conversion, and conservation of solute amount. A two-panel figure shows the mass-to-amount-to-concentration chain and the required aliquots. Predict–observe–explain prompts and a guided doubled-volume exercise make the result checkable. |
| [Chapter 1 Part 2](../Chapter01_Part2.ipynb) | Introduces arrays, tables, rows, columns, axes, replicates, and plotting as different tools with different purposes. Distinguishes first-pass essentials from memory-sharing and file-format details. | An annotated array/row-mean figure makes `axis` and broadcasting visible. A real local measured-solubility analysis audits missing values and units, counts records below a planning threshold, and plots the distribution and threshold dependence. The analysis preserves provenance and does not invent missing pH or temperature. |
| [Chapter 1 Part 3](../Chapter01_Part3.ipynb) | Defines graph, formula, connectivity, isomer, protonation, conformer, descriptor, and InChIKey at their point of use. Moves the complex paclitaxel example after the small-molecule identity lesson, retaining its checked source and assertions. Marks reaction templates and XYZ mapping internals as deeper reference. | A calculated five-pair identity audit shows why raw text, formula, and canonical stereochemical graph answer different questions. A highlighted six-molecule catalog screen contrasts oxygen-containing compounds with a deliberately scoped alcohol SMARTS. The interactive viewer now has a real default-false flag and a static fallback. |
| [Chapter 2 Part 1](../Chapter02_Part1.ipynb) | Bridges graph drawings to positions, displacement vectors, norms, and internal coordinates. Defines distance/angle/dihedral by their questions and atom counts. Separates use of tested geometry helpers from the vector derivations. Removes two uncontextualized legacy torsion images. | Adds a coordinate-derived angle diagram, a Newman-style projection derived from the actual ethane geometry, and a pair-distance-change matrix. The research check verifies that a torsion edit preserves bond lengths while changing nonbonded separations. Reflection/distance limitations and atom mapping remain explicit. The browser viewer is default-false. |
| [Chapter 2 Part 2](../Chapter02_Part2.ipynb) | Explains single points, scans, and local/global minima through the same energy-landscape picture. Makes the catalog of force-field terms a deeper reference. Introduces energy slope, force direction, and stiffness beside a plot. | Adds an explicitly illustrative spring-energy/force figure and three actual independent MMFF94 butane minimizations. The conformer table and plot combine convergence status, relative model energy, torsion, and terminal-carbon distance. The exercise supports selecting starting candidates without claiming populations or a global minimum. The optional animation is now default-false. |
| [Chapter 3](../Chapter03.ipynb) | Introduces the position–force–velocity update, derivative symbols, units/time scales, observables, equilibration, correlation, and uncertainty in plain language. Separates the oscillator/ethane core from protein API details. Returns the unguarded 1000-step protein extension to markdown reference. | Adds exact oscillator potential/kinetic exchange and phase-space views. Uses the actual calculated ethane trajectory for block means and a finite-trace autocorrelation plot. A duplicate-row counterexample shows why a smaller independent-sample standard-error formula is not genuine new information. It makes no convergence or effective-sample-size claim. |

The twelve new figures or molecule grids are generated offline from the displayed data and equations. Their captions and follow-up explanations state what to inspect and which decision is justified. They supplement existing geometry, optimization, and trajectory figures rather than replacing those with decorative illustrations.

## Data and scientific scope

- The NaCl targets and spring parameters are explicit design/model choices. They are not described as measurements. The dilution calculation uses final solution volumes and verifies the back-calculated concentration.
- The solubility application reads the unchanged tracked 1,121-row table. Its target is measured $\log_{10}(S/(1\ \mathrm{mol/L}))$. The current table has zero missing target values; 543 records (48.4%) fall below 1 mmol/L. Counts describe records, not unique structures or independent experiments. Per-record pH, temperature, and measurement details are absent. See the [dataset audit](../datasets/README.md).
- Canonicalization retains charge, isotope, and specified stereochemical information under the current toolkit; it does not reconcile tautomers, salts, missing stereo, or assay conditions. The identity-audit inputs are chosen molecular representations, not measurement records.
- The angle/projection/distance plots use supplied or deliberately edited coordinates. Projection lengths are normalized for readability and are not bond measurements. All distances are calculated from the original coordinates.
- Three butane starts converge under MMFF94 to anti and two gauche geometries here. Gauche results lie about 0.782 kcal/mol above the sampled anti result. These are model potential-energy differences, not measured free energies, solution populations, or proof of a complete conformational search.
- The ethane MD parameters remain an explicitly educational bonded model. Autocorrelation and five short block means describe its two-picosecond trajectory; they do not estimate a reliable correlation time or uncertainty. The solvated-protein example still covers only 0.2 ps and demonstrates preparation/diagnostics, not folding or binding.

## Validation

All six notebooks passed in separate fresh kernels with a **10-second timeout per code cell**, on the native Windows course environment: Python 3.12.14, NumPy 2.5.3, RDKit 2026.03.6, Matplotlib 3.11.2, and the installed OpenMM. Native environment paths and `MKL_THREADING_LAYER=SEQUENTIAL` were set before process startup. Computation used one thread where configured. These timings are environment-specific, not a guarantee for every laptop.

| Notebook | Executed code cells | Sum of code-cell times | Slowest cell | Fresh-kernel wall time |
| --- | ---: | ---: | ---: | ---: |
| Chapter 1 Part 1 | 47 | 0.878 s | 0.761 s | 1.838 s |
| Chapter 1 Part 2 | 32 | 1.777 s | 0.549 s | 2.812 s |
| Chapter 1 Part 3 | 60 | 1.044 s | 0.224 s | 2.076 s |
| Chapter 2 Part 1 | 20 | 1.063 s | 0.337 s | 1.984 s |
| Chapter 2 Part 2 | 24 | 1.744 s | 0.335 s | 2.794 s |
| Chapter 3 | 20 | 10.170 s | 4.908 s | 11.168 s |
| **Total** | **203** | **16.676 s** | **4.908 s** | **22.672 s** |

Checks cover notebook schema and code syntax, existing scientific assertions, dilution conservation, array-axis results, threshold-count monotonicity, the stated identity/query results, unchanged geometry bond lengths, symmetric distance matrices, minimization status/energy, exact oscillator energy conservation, and the duplicate-row counterexample. All twelve new figures/grids were rendered and visually inspected; labels, signs, units, legends, and molecule highlights were checked.

An independent bounded peer review of the new application sections found no substantive chemistry, numerical, or interpretation errors, including the finite-trace autocorrelation and duplicate-row example.

Author-validation copies, extracted figures, and per-cell timing JSON files are in the ignored `outputs/full_review_chapters1_3/validation/` folder. Source outputs were cleared for the parent agent's final course-wide execution. The private validation runner was corrected to close each supplied kernel explicitly before temporary-directory cleanup; the completed rerun exited successfully. No dependency files, tracked datasets, shared validator, or historical validation claims were changed by this review.

All owned notebook markdown uses `$...$` and `$$...$$` for LaTeX; the alternative backslash-parenthesis/bracket wrappers were normalized. Existing code was not rewritten merely to change literal backslashes.

## References checked for the new material

- [NumPy broadcasting documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html): shape compatibility and repeated arithmetic.
- [RDKit getting started](https://www.rdkit.org/docs/GettingStartedInPython.html) and [molecular drawing API](https://www.rdkit.org/docs/source/rdkit.Chem.Draw.rdMolDraw2D.html): graph handling, coordinates, queries, and explicit Cairo rendering.
- [Delaney's original solubility study](https://doi.org/10.1021/ci034243x), [DeepChem's dataset documentation](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html), and the [local provenance record](../datasets/README.md): measured-target identity and export limitations. The publisher blocked automated access to the paper during this pass; the provider documentation and prior exact local/upstream audit were available, and no new claims about measurement conditions were inferred.
- [OpenMM running simulations](https://docs.openmm.org/latest/userguide/application/02_running_sims.html): model setup and simulation controls.
- [Grossfield et al., sampling and uncertainty best practices](https://livecomsjournal.org/index.php/livecoms/article/view/v1i1e5067): correlated trajectories, sampling assessment, and uncertainty interpretation.

[Course contents and setup](../Readme.md)
