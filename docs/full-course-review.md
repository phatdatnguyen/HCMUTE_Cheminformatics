# Full-course teaching review: Chapters 1–12

Completed on **2026-09-16** for graduate students who may be new to cheminformatics. All **35 notebooks** were reviewed and revised, including three new PyTorch Geometric lessons. The [course guide](course-guide.md) gives a learning route, short prerequisite refreshers, worked-project prompts and a graduate assessment rubric. The [main index](../Readme.md) links every notebook and the updated setup instructions.

## What changed

- **A gentler entry to each topic:** definitions and small worked examples precede dense notation; first-pass material is distinguished from deeper derivations and implementation details. Students are prompted to predict, run and interpret results.
- **More explanatory visuals:** molecular depictions, coordinate projections, numerical diagrams, energy and density plots, spectra, reaction analyses, data partitions, learning curves and graph explanations. The final source notebooks contain **188 embedded image outputs**, including existing and new figures and molecular depictions. This counts rendered outputs, not distinct new figures or individual panels.
- **Research applications across the course:** dilution planning, identity and substructure audits, conformer selection, trajectory correlation, basis/grid/model comparisons, isotope shifts, spectral assignment, reaction thermodynamics and kinetics, measured-property prediction, and explanation audits. Each example labels measurements, calculated properties and deliberately illustrative models accurately.
- **Short default calculations:** bounded molecules, scans, trajectories and training loops. Several optional viewers or longer extensions that were running by default are now genuinely optional or presented as reference snippets. The HF torsion scan has 13 points; the DFT teaching grid is smaller and accompanied by an explicit finer-grid comparison.
- **Actual PyTorch Geometric:** Chapter 12 expands from five to eight parts, covering library graph data, batching, layers, regression, classification and `GNNExplainer`. The tested PyG release is pinned in the complete requirements.
- **Consistent mathematical formatting:** notebook Markdown uses `$...$` inline and `$$...$$` for displayed equations. Existing alternate equation delimiters were converted; Python regexes and non-math code were preserved.
- **Reproducible delivery:** successful fresh-kernel outputs are saved into all source notebooks. Data source files remain unchanged, and the course validator now selects all 35 notebooks. Earlier correction reports are retained and clearly marked historical.

## Changes by chapter group

| Chapters | Examples of the new material | Detailed review |
|---|---|---|
| 1–3 | Dilution workflow; measured-solubility audit; molecular identity and alcohol queries; coordinate projections; multistart butane optimization; trajectory correlation. | [Chapters 1–3](full-review-chapters1-3.md) |
| 4–7 | State/probability plots; contraction versus variational coefficients; SCF/geometry loops; water HF/MP2 comparison; density slices and numerical sensitivity; PM7-to-HF geometry check. | [Chapters 4–7](full-review-chapters4-7.md) |
| 8–9 | Fragment cycle; Hückel bond orders; harmonic approximation residuals; D2O isotope shift; spectral broadening; phase-space view; condensation driving force; local saddle; quench decision; reaction audit matrix. | [Chapters 8–9](full-review-chapters8-9.md) |
| 10 | Phenol analogue selection with descriptor space and fingerprint ranking; actual grouped-CV workflow; frozen screening-budget analysis. | [Chapter 10](full-review-chapter10.md) |
| 11 | Gradient/loss geometry; calculated alcohol-mass extrapolation check; hidden-unit construction; measured reaction-yield heatmap; graph reach and symmetry diagrams. | [Chapter 11](full-review-chapter11.md) |
| 12 | Feature-schema corruption; hand-worked messages; solubility decisions; attribution paths; same-graph conformers; actual PyG layers and measured-data experiments. | [Chapter 12](full-review-chapter12.md) |

Every new figure was rendered and visually inspected by its author or reviewer. Layout adjustments addressed legends, axis labels, geometrical scaling and projection readability. Independent read-only reviews covered selected new research applications, the new PyG notebooks and the added Chapter 12 foundations.

## Chapter 12: practical library sequence

- [Part 6](../Chapter12_Part6.ipynb): actual `Data`, `Batch`, `DataLoader`, custom `MessagePassing`, `GCNConv` and `GINEConv`; manual/library forward and gradient agreement, empty-edge graphs, label routing, permutation and batch isolation.
- [Part 7](../Chapter12_Part7.ipynb): measured-solubility GINE regression, short mini-batch training, restored validation checkpoint, matched baselines, residual molecules and a complete reloadable checkpoint.
- [Part 8](../Chapter12_Part8.ipynb): audited BBBP classification, training-only class weighting, validation-selected thresholds, frozen test assessment and a two-seed actual `GNNExplainer` example.

The measured results are useful teaching evidence: descriptor ridge has lower solubility RMSE than GINE on the specified split (1.011 versus 1.089 log units), and Morgan logistic has higher BBBP balanced accuracy (0.793 versus 0.674). The explainer's masks vary between seeds despite similar masked scores. These particular experiments illustrate baseline comparison and interpretation limits; they do not establish a universal model ranking or causal chemical explanation.

## Final validation

Command, run from the activated native Windows course environment:

```sh
python scripts/validate_chapters.py --cell-timeout 10 --inplace
```

**Result: all 35 notebooks and all 571 code cells passed.** Sum of whole-notebook execution times: **187.75 seconds**, about 3.1 minutes. The slowest cell took **8.4227 seconds** (Chapter 6 vibrational calculation). Every notebook used a separate fresh kernel; no earlier notebook's output or session state was required. Cell times include calculations and output rendering; whole-notebook times also include kernel overhead. These measurements describe this machine and software build, not a promise for every laptop.

The final Chapter 12 sequence alone contains **103 executed code cells** and took **39.43 seconds** across eight fresh kernels. Its longest cell was **2.9103 seconds**.

| Notebook | Executed code cells | Whole notebook, seconds | Slowest cell, seconds | Embedded image outputs |
|---|---:|---:|---:|---:|
| [Chapter01_Part1](../Chapter01_Part1.ipynb) | 47 | 1.86 | 0.751 | 1 |
| [Chapter01_Part2](../Chapter01_Part2.ipynb) | 32 | 2.81 | 0.541 | 4 |
| [Chapter01_Part3](../Chapter01_Part3.ipynb) | 60 | 2.09 | 0.221 | 40 |
| [Chapter02_Part1](../Chapter02_Part1.ipynb) | 20 | 1.98 | 0.329 | 7 |
| [Chapter02_Part2](../Chapter02_Part2.ipynb) | 24 | 2.78 | 0.337 | 8 |
| [Chapter03](../Chapter03.ipynb) | 20 | 11.19 | 4.874 | 7 |
| [Chapter04](../Chapter04.ipynb) | 15 | 2.95 | 0.572 | 4 |
| [Chapter05](../Chapter05.ipynb) | 23 | 8.61 | 2.981 | 7 |
| [Chapter06](../Chapter06.ipynb) | 14 | 21.77 | 8.423 | 6 |
| [Chapter07](../Chapter07.ipynb) | 11 | 4.12 | 1.100 | 5 |
| [Chapter08_Part1](../Chapter08_Part1.ipynb) | 10 | 3.64 | 0.835 | 5 |
| [Chapter08_Part2](../Chapter08_Part2.ipynb) | 10 | 3.38 | 0.829 | 3 |
| [Chapter08_Part3](../Chapter08_Part3.ipynb) | 12 | 2.75 | 0.992 | 4 |
| [Chapter08_Part4](../Chapter08_Part4.ipynb) | 11 | 2.53 | 0.531 | 5 |
| [Chapter08_Part5](../Chapter08_Part5.ipynb) | 10 | 6.83 | 2.805 | 3 |
| [Chapter08_Part6](../Chapter08_Part6.ipynb) | 9 | 3.48 | 1.083 | 3 |
| [Chapter08_Part7](../Chapter08_Part7.ipynb) | 11 | 8.69 | 3.596 | 2 |
| [Chapter09_Part1](../Chapter09_Part1.ipynb) | 10 | 2.41 | 0.593 | 5 |
| [Chapter09_Part2](../Chapter09_Part2.ipynb) | 16 | 24.77 | 4.935 | 4 |
| [Chapter09_Part3](../Chapter09_Part3.ipynb) | 11 | 3.42 | 0.754 | 5 |
| [Chapter09_Part4](../Chapter09_Part4.ipynb) | 10 | 1.81 | 0.436 | 4 |
| [Chapter10_Part1](../Chapter10_Part1.ipynb) | 13 | 2.19 | 0.519 | 3 |
| [Chapter10_Part2](../Chapter10_Part2.ipynb) | 17 | 5.11 | 1.201 | 5 |
| [Chapter11_Part1](../Chapter11_Part1.ipynb) | 12 | 4.14 | 2.082 | 4 |
| [Chapter11_Part2](../Chapter11_Part2.ipynb) | 13 | 4.38 | 2.102 | 4 |
| [Chapter11_Part3](../Chapter11_Part3.ipynb) | 16 | 5.55 | 2.909 | 4 |
| [Chapter11_Part4](../Chapter11_Part4.ipynb) | 11 | 3.08 | 1.229 | 5 |
| [Chapter12_Part1](../Chapter12_Part1.ipynb) | 17 | 3.14 | 1.435 | 3 |
| [Chapter12_Part2](../Chapter12_Part2.ipynb) | 11 | 3.86 | 2.146 | 3 |
| [Chapter12_Part3](../Chapter12_Part3.ipynb) | 16 | 6.23 | 2.910 | 4 |
| [Chapter12_Part4](../Chapter12_Part4.ipynb) | 13 | 4.67 | 2.163 | 4 |
| [Chapter12_Part5](../Chapter12_Part5.ipynb) | 13 | 4.66 | 2.143 | 5 |
| [Chapter12_Part6](../Chapter12_Part6.ipynb) | 9 | 4.41 | 2.523 | 3 |
| [Chapter12_Part7](../Chapter12_Part7.ipynb) | 12 | 6.34 | 2.501 | 4 |
| [Chapter12_Part8](../Chapter12_Part8.ipynb) | 12 | 6.12 | 2.475 | 5 |
| **Total** | **571** | **187.75** | **8.423** | **188** |

The machine-readable run is saved locally at `outputs/validation/report-full-course.json`; `outputs/validation/report.json` contains the same final integration run. Generated output directories are intentionally ignored by Git; the table above is the durable execution record. Verified visual and textual results are embedded in the source notebooks.

### Environment and additional checks

The tested environment used Python 3.12.14, RDKit 2026.03.6, NumPy 2.5.3, SciPy 1.18.1, pandas 2.3.3, Matplotlib 3.11.2, OpenMM 8.6.1, Psi4 1.11, MOPAC 23.2.5, scikit-learn 1.8.0, PyTorch 2.11.0+cpu and PyG 2.8.0.post1. The numerical examples use one CPU thread where configured, with `MKL_THREADING_LAYER=SEQUENTIAL` selected before imports. `pip check` reports no broken requirements.

Notebook schema and code syntax, sequential saved execution counts, absence of saved error outputs, local navigation links, equation delimiters, and dataset SHA-256 values were checked separately. Scientific assertions include conservation/normalization, geometry and energy checks, gradient identities, proper data partitions, training-only transformations, best-checkpoint restoration and model reload agreement. Whitespace validation with `git diff --check` passes.

The following source hashes match the unchanged bundled datasets:

| Dataset | SHA-256 |
|---|---|
| Solubility.csv | `3fedd5ad80f9231bd331929ba0943a117d0d6ee3f75eda9a27fab9b4ab974ad5` |
| Lipophilicity.csv | `657596792d4980fc196e7833a98bf1373dc0b4f8f52ece0f905f59abfac44b80` |
| BBBP.csv | `d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662` |
| BuchwaldHartwigReactionYield.csv | `ff4ff0336e2f244e7b01058499634aea9714470d518af0044d355163c5a39972` |

## Scope students should retain

Short trajectories do not establish equilibrium, isolated low-level calculations do not certify experimental accuracy, and a local minimum does not establish a global minimum. Model energies, formation enthalpies, free energies and measurements retain their own definitions and reference states. Numerical convergence and scientific accuracy are separate questions.

The data examples preserve known limitations: heterogeneous or incomplete measurement conditions, duplicate/contradictory observations, small validation groups, restricted chemical subsets and deliberately bounded training. A held-out score applies to its stated population and protocol. Post-assessment plots are labeled and are not used to retune models or remove failures. Graph features omit information when explicitly declared, and explanation plots do not establish causality.

Default calculations run offline after installation, using embedded examples and tracked data. Optional 3D viewers can require browser access to their JavaScript library; static figures remain available. Extended research protocols are described without adding long default calculations.

The next teaching step is to use the [course guide's projects and rubric](course-guide.md#short-research-projects-using-the-bundled-material), asking students for a reproducible scientific conclusion supported by figures and checks, including when a baseline wins or a proposed explanation fails.
