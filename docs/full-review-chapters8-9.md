# Beginner-focused review of Chapters 8 and 9

This pass improves all eleven notebooks for graduate students encountering molecular computation for the first time. It preserves the existing verified calculations, adds a short prerequisite bridge to each lesson, and connects a concrete research question to a small worked analysis, a visual, and an explicitly limited conclusion. Existing exercises and answers remain. Added self-checks address interpretation, not just code syntax.

All notebook Markdown uses `$...$` for inline mathematics and `$$...$$` for display mathematics. Advanced response-matrix algebra in Part 8.6 is placed in a clearly labeled expandable section; the core spectrum interpretation can be read first.

## Changes by notebook

| Notebook | Prerequisite bridge | New analysis and visual | Research interpretation |
|---|---|---|---|
| [8.1](../Chapter08_Part1.ipynb) | Single points, SCF, density, dipole, and electrostatic potential answer different questions. | Sensitivity bars compare changing a large dielectric constant with changing the cavity radius by 10%, using the existing calculated water dipole. | Test assumptions that dominate the predicted response before interpreting solvent differences. These remain fixed-dipole cavity contributions, not measured solvation energies. |
| [8.2](../Chapter08_Part2.ipynb) | Charge, radical character, alpha/beta spin, and heterolytic versus homolytic fragments. | A fragment-energy cycle connects homolysis, H ionization, OH electron attachment, and heterolysis using the same computed species. | Cycle closure catches accounting errors; it cannot repair the known HF electron-affinity error or turn gas electronic energies into pKa. |
| [8.3](../Chapter08_Part3.ipynb) | AO building blocks, MO phase, HOMO/LUMO, and the sigma/pi distinction. | Occupied Hückel density matrices produce a labeled bond-width plot of neighboring pi bond orders in butadiene and hexatriene. | Equal couplings do not imply equal occupied bond orders; this motivates an explicitly separate geometry study without pretending bond order is a measured length. |
| [8.4](../Chapter08_Part4.ipynb) | A molecular energy map, slopes, curvature, and the limitations of a two-coordinate slice. | Fit a quadratic model on the central 3 × 3 points of the existing MMFF94 grid; plot its extrapolated cut and signed residual map. | A local harmonic approximation can fail for larger deformations. The residual is relative to this force field, not experimental error or a full vibrational Hessian. |
| [8.5](../Chapter08_Part5.ipynb) | Coupled molecular springs, mass, wavenumber, mode displacements, and IR activity. | Reuse the actual water Cartesian Hessian with deuterium masses; verify recovery of Psi4 H2O frequencies and show the H2O/D2O shift. | Isotope shifts can test assignments without changing the clamped-nuclei electronic potential. The exercise predicts harmonic positions, not D2O intensities or experimental bands. |
| [8.6](../Chapter08_Part6.ipynb) | Photon energy versus wavelength; excitation roots versus orbitals; transition energy versus visibility. | Broaden the same four TDHF roots with three normalized Gaussian widths and identify the lowest root, brightest transition, and profile maximum. | Peak assignment needs intensity and state character. Broadening does not change the states or their included integrated oscillator strength. |
| [8.7](../Chapter08_Part7.ipynb) | Distinguish SCF iterations, optimization iterations, and physical MD time steps. | A bond-length/velocity phase portrait, radial kinetic-energy identity, and coarse/fine trajectory comparison at matching times. | Positions do not determine motion without velocities. Timestep convergence is distinct from potential accuracy or ensemble sampling. |
| [9.1](../Chapter09_Part1.ipynb) | Enthalpy, entropy, Gibbs energy, reaction differences, and the meaning of a standard state. | Add NIST liquid-water thermochemistry to derive a 298.15 K coexistence pressure; plot the condensation driving force against water partial pressure. | The allowed phases matter. An equilibrium driving force does not predict nucleation speed; the derived pressure uses ideal-vapor/pure-liquid assumptions. |
| [9.2](../Chapter09_Part2.ipynb) | Stationary points, minima, saddles, imaginary frequencies, and IRC versus time. | Plot a local quadratic saddle and its stable/unstable cuts using the actual computed squared-frequency ratio. | A one-dimensional maximum is insufficient. Gradient, complete internal-mode classification, and endpoint connection give separate evidence. |
| [9.3](../Chapter09_Part3.ipynb) | Concentration, rate, rate constant, barrier, and population evolution. | Reuse the reversible branch trajectory to compare conversion, desired-product yield, and selectivity; identify the largest sampled P amount under an ideal-quench thought experiment. | High product selectivity can coexist with low yield. The model parameters are invented teaching values, and an isolated experimental yield needs additional evidence. |
| [9.4](../Chapter09_Part4.ipynb) | Specific reaction records, atom correspondence, graph templates, and mechanisms. | A pass/fail/not-applicable matrix compares five independent chemical-bookkeeping checks across the valid and deliberately damaged records. | A parser success is insufficient; multiple defects can coexist. Passing all checks still does not prove feasibility, mechanism, or yield. |

## References checked for the new material

- [NIST water gas and liquid thermochemistry](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=3): CODATA liquid formation enthalpy −285.830 ± 0.040 kJ/mol and entropy 69.95 ± 0.03 J/(mol K), paired with the existing gas values at 298.15 K. The ideal-phase derivation gives approximately 3.17 kPa, not a newly measured vapor pressure.
- [NIST hydrogen isotope masses](https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=H): deuterium atomic mass 2.01410177812 u, used only in nuclear mass weighting.
- [Psi4 frequency and Hessian documentation](https://psicode.org/psi4manual/master/freq.html): Cartesian Hessian units, wavefunction access, normal-mode arrays, and translation/rotation/vibration classification.
- [Hosoya and colleagues, electron density and bond order](https://publications.iupac.org/pac/pdf/1983/pdf/5502x0269.pdf): the occupied-orbital electron-density and Coulson pi bond-order convention. This convention is confined to the specified orthonormal pi model.
- Existing primary references for Onsager response, HF/SCF, MMFF94, time-dependent SCF, transition-state theory, NIST gas thermochemistry, and RDKit reaction handling remain linked where used. The new analyses of already computed arrays do not introduce a new engine or external dataset.

Sources checked during this review on 16 September 2026. Reference values are embedded so default execution remains offline.

## Validation and runtime

All eleven notebooks passed **120 code cells** in separate fresh kernels with a 10-second per-cell timeout, one CPU thread, and no network access needed by notebook code. The new analyses add **zero quantum-chemistry engine calls**. The private runner was copied from the shared validator with a separate report directory; shared validation files were not modified. All eleven new figures were visually inspected for readable labels, units, overlap, and agreement with the stated conclusions.

| Notebook | Code cells | Whole notebook, seconds | Slowest cell, seconds |
|---|---:|---:|---:|
| Chapter08_Part1 | 10 | 4.14 | 1.222 |
| Chapter08_Part2 | 10 | 3.45 | 0.835 |
| Chapter08_Part3 | 12 | 2.83 | 1.037 |
| Chapter08_Part4 | 11 | 2.50 | 0.533 |
| Chapter08_Part5 | 10 | 7.05 | 2.871 |
| Chapter08_Part6 | 9 | 3.56 | 1.127 |
| Chapter08_Part7 | 11 | 9.12 | 3.850 |
| Chapter09_Part1 | 10 | 2.42 | 0.582 |
| Chapter09_Part2 | 16 | 25.22 | 5.004 |
| Chapter09_Part3 | 11 | 3.39 | 0.747 |
| Chapter09_Part4 | 10 | 1.73 | 0.433 |

Times include kernel work and rendered outputs on the review machine, not a guarantee for other hardware. Part 9.2 remains longer in total because its existing transition-structure, short IRC, and endpoint calculations are deliberately separate short cells. New mathematical checks include energy-cycle closure, density electron counts, isotope eigenvalue separation and recovery of reference frequencies, line-area conservation, kinetic-energy reconstruction, and yield–conversion–selectivity consistency.

The tested environment used Python 3.12, Psi4 1.11, RDKit 2026.03.6, NumPy 2.5.3, SciPy 1.18.1, pandas 2.3.3, and Matplotlib 3.11.2 with `MKL_THREADING_LAYER=SEQUENTIAL`. Development records and executed copies are under ignored `outputs/full_review_8_9_validation/`. Source notebooks are handed back without stale outputs for the integrator's final course-wide execution. After the full run, minor Markdown clarifications and an isotope-figure legend repositioning were checked separately; the legend was re-rendered from the generated numerical table without repeating quantum calculations.
