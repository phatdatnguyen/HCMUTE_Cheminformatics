# Chapter 8: molecular properties with short, checked examples

> Historical correction record. The later [full-course teaching review](full-course-review.md) supersedes the notebook counts, dependency scope, and runtime totals below.


## Scope and design

All seven parts of Chapter 8 have been revised. The chapter retains its progression through energies and molecular fields, charged species, orbitals, potential-energy surfaces, vibrations, electronic spectra, and molecular dynamics. The examples use smaller systems and explicitly defined models so the required calculations stay short.

Each notebook runs independently in the existing course environment. Molecular inputs are embedded, figures use Matplotlib, and no previous notebook output, internet download, external viewer, or legacy `utils.py` helper is required. There are no hidden optional cells that launch large calculations. Research extensions are described as questions and methodological considerations.

## Changes to the examples and scientific content

| Part | Replacement example | Main corrections and checks |
| --- | --- | --- |
| [1: Energies and molecular fields](../Chapter08_Part1.ipynb) | Three RHF/STO-3G water single points, small density/ESP plane grids, and an analytical Onsager cavity model | Meaningful energy references; physical dipole direction and atomic-unit/Debye conversion; dipole checked against AO integrals; total electron count; density versus ESP; plane contours versus 3D surfaces; clearly delimited solvent-response model |
| [2: Ions and radicals](../Chapter08_Part2.ipynb) | Small HF calculations for OH+, OH, OH−, water, and H | Charge and multiplicity; diffuse functions; actual density and UHF spin diagnostics; explicit electron/proton bookkeeping; ionization versus attachment, homolysis, and proton loss; an independently checked energy cycle; electronic energies distinguished from solution pKa |
| [3: Molecular orbitals](../Chapter08_Part3.ipynb) | Hückel models of two polyenes and actual RHF/STO-3G H2 orbitals | Analytic/matrix eigenvalue agreement; phase and nodes; overlap-metric normalization; occupations; orbital energies distinguished from total energies and optical transitions; actual orbital values on a small spatial slice |
| [4: Potential-energy surfaces](../Chapter08_Part4.ipynb) | Fast MMFF94 ethanol scans and a small hydrogen-only relaxation comparison | Every grid point copies the same reference; measured scan coordinates and consistent array axes; common energy zero; explicit unrelaxed versus partially relaxed definitions; fixed coordinates, parameter coverage, and convergence checked; potential versus free energy |
| [5: Vibrations and thermochemistry](../Chapter08_Part5.ipynb) | Actual HF/STO-3G water optimization and finite-difference frequencies, IR intensities, and thermochemistry | Gradient and internal-mode checks; imaginary modes; normal-mode arrows; area-normalized broadening; intensity versus transmittance; ZPVE counted once; entropy units; ideal-gas RRHO at 1 bar; reference-state conversion |
| [6: Electronic excitation](../Chapter08_Part6.ipynb) | Four formaldehyde singlet transitions using CIS and full TDHF | Correct response API and retained JK object; converged reference and explicit response settings; TDA/CIS/TDHF/TDDFT distinctions; vertical excitation and oscillator strengths; energy/wavelength conversion; normalized profiles and the spectral-density Jacobian |
| [7: Quantum forces and dynamics](../Chapter08_Part7.ipynb) | Two real RHF/STO-3G H2 BOMD trajectories over 4 fs, plus one PBE force comparison | Gradient sign verified by finite differences; atomic-unit positions, masses, and time; velocity Verlet; matching energy/force geometry; momentum and center-of-mass conservation; timestep convergence; checked multi-frame XYZ export |

The original batches of substituted-aromatic optimizations, dense DFT surfaces, larger DFT Hessians, and acetone force loops are replaced. The density/ESP and MO visualizations evaluate the calculated electronic fields directly on small grids. The solvent section is explicitly an analytical electrostatic model. The ethanol surface is explicitly a force-field calculation. These labels keep the computational method and the teaching claim aligned.

## Validation and runtime

On 2026-09-15, all **66 code cells across seven notebooks** passed serial execution in separate fresh kernels with a **10-second timeout per cell**. The source notebooks contain the verified outputs. Total notebook execution time, including kernel startup and shutdown, was **30.30 seconds**; the slowest individual cell took **3.561 seconds**.

| Part | Executed code cells | Notebook time | Slowest cell | Result |
| --- | ---: | ---: | ---: | --- |
| 1 | 9 | 4.02 s | 1.176 s | Passed |
| 2 | 9 | 3.17 s | 0.827 s | Passed |
| 3 | 11 | 2.73 s | 1.036 s | Passed |
| 4 | 10 | 2.17 s | 0.528 s | Passed |
| 5 | 9 | 6.74 s | 2.734 s | Passed |
| 6 | 8 | 3.19 s | 1.091 s | Passed |
| 7 | 10 | 8.28 s | 3.561 s | Passed |

Checks also covered notebook schema, code syntax, regenerated outputs without errors, numerical assertions, local navigation, and rendered figures. All seven parts received a scientific review independent of their author.

Use the activated course environment:

```sh
python scripts/validate_chapters.py --chapter8 --cell-timeout 10 --inplace
```

The [validator](../scripts/validate_chapters.py) executes the seven parts in separate fresh kernels, stops a notebook on a cell error or timeout, and saves verified outputs only after a successful notebook run. Its local `outputs/validation/report.json` records software versions, notebook runtimes, and zero-based cell indices with kernel busy-to-idle times, which include calculation and output rendering. The 10-second limit is a validation guard; observed times depend on hardware, system load, and software build.

The setup uses the existing native Windows course environment: Python 3.12.14, Psi4 1.11, NumPy 2.5.3, SciPy 1.18.1, pandas 2.3.3, RDKit 2026.3.6, and Matplotlib 3.11.2. Psi4 calculations use one thread and 512 MiB. Chapter 8 adds no package dependencies; [requirements-chapters1-8.txt](../requirements-chapters1-8.txt) includes the earlier requirements. Activate the environment before launching Jupyter or validation on Windows so the native library paths are available. See the [setup guide](../Readme.md#set-up-python).

## Interpretation and remaining limits

- **Numerical success is distinct from chemical accuracy.** The deliberately small HF bases do not provide quantitative attachment energies, vibrational benchmarks, or spectroscopy. The OH attachment example explicitly discusses the model's incorrect electron-affinity sign against a cited experimental reference.
- **State and reference conventions matter.** Raw total energies of unlike formulas are not a stability scale. Electrons/protons and all atoms must balance. Gas-phase electronic differences do not supply solution free energies or pKa values.
- **The solvent example is a limited model.** The fixed-dipole spherical cavity omits molecular shape, solute polarization, specific solvent interactions, non-electrostatic terms, and full thermodynamic conventions. Its radii are illustrative.
- **Orbital and spectral pictures need precise labels.** Orbital phase is not electrical charge; a plane slice is not an isosurface; orbital eigenvalue gaps are not optical transitions. Normal-mode arrows use an arbitrary display scale. Broadening widths are selected for visualization, not predicted lifetimes or linewidths.
- **Spectral intensity is not transmittance.** The IR profile preserves integrated calculated intensity, and the electronic profile preserves summed oscillator strength. Neither is calibrated absorbance. The wavelength Jacobian applies to a density per unit horizontal variable, not automatically to every experimental curve.
- **Thermochemistry assumes a model and standard state.** Water RRHO results refer to an ideal-gas species at 298.15 K and 1 bar, not liquid water. Anharmonicity, hindered rotations, conformer ensembles, and solvation need additional treatment where relevant.
- **The dynamics is intentionally short.** The 4 fs H2 trajectories test the implementation and timestep sensitivity. They do not establish equilibrium, spectra, reaction rates, or quantum nuclear effects. Both use one fixed HF force model; the PBE calculation is a separate single-point force comparison.
- **Later course material has separate review records.** Subsequent reviews cover [Chapter 9](chapter9-review.md) and [Chapters 10–11](chapters10-11-review.md). Legacy shared helpers remain outside the revision. The [Chapters 1–3](chapters1-3-review.md) and [Chapters 4–7](chapters4-7-review.md) notes preserve their earlier validation records.

## Sources

Primary references and official documentation are linked beside the relevant explanations in each notebook. Core sources include:

- [IUPAC electric dipole moment](https://doi.org/10.1351/goldbook.E01929), [ionization energy](https://doi.org/10.1351/goldbook.I03199), and [electron affinity](https://doi.org/10.1351/goldbook.E01977).
- [Psi4 SCF](https://psicode.org/psi4manual/master/scf.html), [molecular coordinates](https://psicode.org/psi4manual/master/api/psi4.core.Molecule.html), [vibrations](https://psi4.github.io/psi4docs/master/freq.html), [thermochemistry](https://psi4.github.io/psi4docs/master/thermo.html), and [time-dependent response](https://psi4.github.io/psi4docs/master/tdscf.html).
- [RDKit molecular transforms](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolTransforms.html) and [force-field helpers](https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html).
- [NIST OH thermochemistry](https://webbook.nist.gov/cgi/cbook.cgi?ID=C3352576&Mask=460&Units=CAL) and [physical constants](https://pml.nist.gov/cuu/Constants/).
- [IUPAC Beer–Lambert law](https://goldbook.iupac.org/terms/view/B00626) and [standard pressure](https://goldbook.iupac.org/terms/view/S05921).
