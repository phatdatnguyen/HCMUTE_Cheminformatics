# Chapter 9: thermodynamics, mechanisms, kinetics, and reaction records

> Historical correction record. The later [full-course teaching review](full-course-review.md) supersedes the notebook counts, dependency scope, and runtime totals below.


## Scope and teaching sequence

Chapter 9 now contains four independent notebooks. The original two parts are revised, and two new parts connect reaction energetics to kinetic modeling and practical cheminformatics. All required inputs are embedded; the notebooks run offline in the existing course environment. They use static figures and do not rely on untracked structures, previous notebook outputs, browser viewers, or legacy `utils.py` helpers.

| Part | Worked example | Main changes and checks |
| --- | --- | --- |
| [1: Reaction thermodynamics](../Chapter09_Part1.ipynb) | Evaluated NIST gas-phase H₂/O₂/H₂O data; Diels–Alder molecular bookkeeping | Consistent formation-energy references; balanced stoichiometry; entropy units; dimensionless activities, Q, and K; pressure effects; invariant driving force under a change of standard-state convention |
| [2: Transition states and paths](../Chapter09_Part2.ipynb) | Actual three-atom HCN/HNC calculations using RHF/STO-3G | Unconstrained saddle optimization; small gradient and one imaginary internal mode; mode displacement; both short IRC branches followed by unconstrained endpoint refinement and vibrational classification; explicitly electronic barrier references |
| [3: Reaction rates and selectivity — new](../Chapter09_Part3.ipynb) | Small reversible and competing-path kinetic models | Eyring units and barrier sensitivity; analytic/numerical agreement; conserved concentrations; detailed balance; catalytic rates with unchanged equilibrium; finite-time selectivity versus equilibrium composition |
| [4: Reaction representations — new](../Chapter09_Part4.ipynb) | Mapped halide substitution and bounded RDKit product enumeration | Reaction SMILES versus reaction SMARTS; heavy-atom mapping, hydrogen/isotope/charge balance; net graph edits; explicit product-charge changes; sanitization; duplicate matches; stereochemical inversion and unspecified stereo; provenance |

Each part includes learning objectives, explanations beside the code, exercises with answers, and primary references. The final part summarizes which evidence answers questions about equilibrium, paths, rates, and molecular graph transformations.

## Corrections to the original examples

### Reaction thermodynamics

The original Diels–Alder example launched larger DFT optimizations and frequency calculations for three species. The replacement completes the numerical thermodynamics lesson using a small, cited reference-data table. Diels–Alder connectivity and formula checks remain, together with a discussion of the conformer, stationary-point, phase, and statistical-mechanical work needed for a defensible molecular calculation.

The revision distinguishes formation enthalpies from quantum-chemical total energies, retains the temperature denominator in entropy units, and checks reaction balance before energy arithmetic. It distinguishes the standard reaction Gibbs energy from the driving force at the current composition. Its gas-pressure and concentration-reference calculations describe the same ideal-gas model; a standard-state conversion is not presented as a solvation calculation. Water's hypothetical gas standard state at 298.15 K is distinguished from stable liquid water.

### Transition states and connectivity

The original SN2 notebook depended on untracked transition-structure files, froze hydrogen coordinates in its transition-state search, then replaced the optimized structure with a different file before analysis. It also mixed separated-reactant and encounter-complex references, mislabeled chemical species, and treated long optimization/IRC output as if it were a physical trajectory.

The revised computational example uses HCN/HNC to make the full verification workflow small enough for class. Its starting geometry is embedded and attributed to an official Psi4 example. All stationary structures use one explicit electronic-structure model. Frequencies are calculated by finite differences of analytic gradients, supported by the tested Windows build. Short IRC segments and subsequent unconstrained endpoint optimizations are identified separately. Their iteration histories have no physical timestep.

The SN2 reaction remains a useful conceptual comparison for reference energies and appears as an explicit mapped graph example in Part 4. Chloride and bromide are identified correctly, and the byproduct is retained for net atom and charge balance.

### Rates, selectivity, and representations

Part 3 introduces rates as a separate modeling problem. It distinguishes electronic barriers, activation Gibbs energies, and Arrhenius activation energies; accounts for concentration units in bimolecular Eyring coefficients; and enforces equilibrium consistency in both directions. The competing-path model distinguishes an initially favored product from the equilibrium mixture and separates product composition from conversion and isolated yield.

Part 4 makes reaction representation a practical notebook topic. Its auditor documents the limited class of complete mapped net equations that it accepts. Partial records are flagged without declaring the underlying reported chemistry impossible. Product generation is bounded and checked for sanitization and atom/charge conservation. A tertiary-halide match illustrates that a graph rule does not establish an SN2 mechanism. Stereo examples preserve unspecified information instead of inventing an enantiomeric outcome.

## Validation and runtime

On 2026-09-15, all **43 code cells across four notebooks** passed serial execution in separate fresh kernels with a **10-second timeout per cell**. The source notebooks contain the verified outputs. Total notebook time, including kernel startup and shutdown, was **38.41 seconds**; the slowest individual cell took **5.406 seconds**.

| Part | Executed code cells | Notebook time | Slowest cell | Result |
| --- | ---: | ---: | ---: | --- |
| 1 | 9 | 2.66 s | 0.952 s | Passed |
| 2 | 15 | 31.39 s | 5.406 s | Passed |
| 3 | 10 | 3.05 s | 0.747 s | Passed |
| 4 | 9 | 1.31 s | 0.294 s | Passed |

The checks include:

- NIST water-formation ΔrG° = −228.581879 kJ/mol at 298.15 K, consistent with the separately tabulated rounded value; reaction reversal/scaling and pressure/concentration descriptions agree.
- A TS with one imaginary internal wavenumber near 1248.634i cm⁻¹ and two positive modes; HCN and HNC endpoints each have four positive internal modes. All three stationary structures have maximum Cartesian gradient components below 10⁻⁵ Eh/bohr in this run.
- Four converged IRC points in each direction, decreasing electronic energies away from the saddle, and subsequent refinement to distinct HCN and HNC minima. The electronic forward/reverse barriers are approximately 289.745/208.954 kJ/mol and differ by the 80.791 kJ/mol model reaction energy.
- Numerical reversible concentrations agreeing with the analytic solution to about 1.4 × 10⁻¹³ M, conserved concentrations, detailed balance, and consistent bimolecular concentration-unit conversion.
- Rejection of incomplete or inconsistent mapped net equations; atom/isotope/charge conservation in generated product bundles; duplicate-match handling; inversion for both specified enantiomers and retained uncertainty for an unspecified center.

All four parts received an independent scientific review. Notebook schema, Python syntax, local navigation, regenerated outputs, and rendered figures were also checked. Fresh-kernel testing exposed a Jupyter-specific RDKit image-saving mismatch in Part 1; explicit Cairo PNG generation fixed it before the final successful run.

From the activated course environment:

```sh
python scripts/validate_chapters.py --chapter9 --cell-timeout 10 --inplace
```

The [validator](../scripts/validate_chapters.py) selects all four parts with `--chapter9`. It saves a source notebook only after that notebook succeeds, and records versions and per-cell kernel busy-to-idle times in `outputs/validation/report.json`. These times include calculations and output rendering. The final review report is also retained locally as `outputs/validation/report-chapter9.json`. The timeout is a validation guard; observed times vary with hardware, system load, and software build. There are no long optional calculation cells in Chapter 9.

The tested native Windows environment used Python 3.12.14, Psi4 1.11, OptKing 0.5.0, NumPy 2.5.3, SciPy 1.18.1, pandas 2.3.3, RDKit 2026.3.6, and Matplotlib 3.11.2. Psi4 uses one thread and 512 MiB. Chapter 9 adds no dependencies: [requirements-chapters1-9.txt](../requirements-chapters1-9.txt) includes the earlier requirements, and [environment.yml](../environment.yml) installs the quantum engines. See the [setup instructions](../Readme.md#set-up-python).

## Interpretation and remaining limits

- **The examples have different evidence types.** Part 1 analyzes evaluated reference data. Part 2 performs actual quantum-chemical calculations. Part 3 assigns explicitly invented free-energy parameters. Part 4 generates molecular graphs. Their records preserve these distinctions; none is presented as an experimental reaction-yield prediction.
- **RHF/STO-3G is a deliberately small model.** A numerically verified saddle and endpoints on this surface do not provide a quantitative experimental barrier, temperature-dependent rate, or solution mechanism. Correlation, basis convergence, nuclear quantum effects, and the relevant environment require further work.
- **Short IRC segments are local path evidence.** Following each direction and refining the endpoints supports connection to two basins on the selected surface. The stored samples do not constitute a densely converged IRC all the way to both minima, and an optimizer's microsteps are not the same as converged IRC points. A complete mechanism can include other saddles and intermediates.
- **Electronic and free-energy barriers answer different questions.** Part 2 does not turn its electronic barrier into an Eyring rate. Part 3 teaches that conversion with explicitly supplied activation free energies and assumptions. Standard states and reference species must match the requested rate law.
- **The kinetic networks are deliberately small.** They assume closed, ideal dilute systems with elementary mass-action behavior and fixed temperature. Catalysis is represented as an effective additional route at fixed catalyst activity. Binding, transport, persistent driving, and product removal need expanded models.
- **A graph transformation is not a mechanism predictor.** Atom mapping, balance, and valid valence do not establish chemical feasibility, stereoselectivity under real conditions, or reaction yield. The identity keys in Part 4 preserve molecular components and specified stereochemistry but do not normalize tautomer or protonation states.
- **Later chapters have a separate review record.** The subsequent [Chapters 10–11 review](chapters10-11-review.md) covers their six notebooks. Legacy shared helpers remain outside the revision. Earlier verification records remain in the [Chapters 1–3](chapters1-3-review.md), [Chapters 4–7](chapters4-7-review.md), and [Chapter 8](chapter8-review.md) notes.

## Sources

Sources are linked beside the relevant lesson content. Core references include:

- NIST Chemistry WebBook entries for [hydrogen](https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Units=SI&Mask=1), [oxygen](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Units=SI&Mask=1), and [water](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=1), plus the [NIST-JANAF water table](https://janaf.nist.gov/tables/H-064.html).
- [IUPAC recommendations on standard quantities and activities](https://doi.org/10.1351/pac199466030533) and [Psi4 thermochemistry](https://psicode.org/psi4manual/master/thermo.html).
- [Psi4 geometry optimization](https://psicode.org/psi4manual/master/optking.html), [vibrational analysis](https://psicode.org/psi4manual/master/freq.html), and the [official HCN/HNC IRC example](https://github.com/psi4/psi4/blob/master/samples/opt-irc-2/input.dat).
- [Eyring's rate-theory paper](https://doi.org/10.1063/1.1749604), [IUPAC Green Book](https://iupac.qmul.ac.uk/bibliog/GreenBook3rd2pt2008.pdf), and [SciPy integration documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
- [RDKit reaction API](https://www.rdkit.org/docs/source/rdkit.Chem.rdChemReactions.html), [reaction SMARTS and stereochemistry](https://www.rdkit.org/docs/RDKit_Book.html#reaction-smarts), [explicit product-property changes](https://greglandrum.github.io/rdkit-blog/posts/2025-04-26-specifying-changes-in-reactions.html), and [Daylight reaction SMILES](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html).

[Course setup and contents](../Readme.md)
