# Beginner-focused review: Chapter 11

Reviewed on 2026-09-16 for graduate students beginning cheminformatics and machine learning. All four notebooks were read in full. The existing bounded training protocols, baseline comparisons, selected-checkpoint restoration, measured-data provenance, and saved-model checks were preserved. Changes make the representations, calculations, and research decisions visible before presenting implementation details.

## Changes by notebook

| Notebook | Beginner support | New visual and research work |
| --- | --- | --- |
| [Part 1: linear units and gradients](../Chapter11_Part1.ipynb) | Defines feature, target, weight, bias, loss, and learning rate; introduces symbols with a table and gives a first-pass route. Separates the input–prediction–error loop from matrix derivatives, rank, and checkpoint details. | A two-parameter slice of the existing one-row loss shows the continuum of equally fitting solutions and a checked downhill step. Equal axis scaling makes the gradient direction geometrically faithful. A separate actual RDKit calculation checks the molar masses of eight straight-chain alcohols, compares a predeclared five-row OLS fit with three held-out compositions, and verifies the exact atomic-weight formula. |
| [Part 2: nonlinear networks](../Chapter11_Part2.ipynb) | Explains hidden units, activations, architecture, interpolation, and extrapolation. Uses predict–observe–explain prompts before numerical examples. | Expands the activation figure to include slopes, saturation, and the explicit ReLU autograd convention at zero. Three hand-specified ReLU contributions build an exactly checked triangular response. A frozen selected model is compared with the known synthetic relationship outside the training range, making the extrapolation limitation visible without another training or selection loop. |
| [Part 3: chemical-data experiment](../Chapter11_Part3.ipynb) | Gives a scientific workflow table and defines scaffold, baseline, descriptor, and generalization. Connects the reaction table's ligand/additive/base/aryl-halide roles to the experimental context. | Shows the actual 360/42/198-row scaffold allocation, including the single acyclic group in testing. A second figure displays actual measured reaction yields on an additive-by-aryl-halide grid for one fixed ligand/base pair chosen independently of yield. Held-out aryl columns are marked; an ID-to-SMILES key and the exact panel are saved. The exercise distinguishes scattered missing combinations from unseen substrates. |
| [Part 4: graph neural networks](../Chapter11_Part4.ipynb) | Defines embedding, message, update, readout, equivariance, and invariance before the equations; labels the detailed batching and geometry mechanics as deeper material. Fixes an obvious corrupted navigation label. | Three graph panels show oxygen information's possible zero/one/two-round reach in ethanol. Actual untrained node-embedding heatmaps show row equivariance, while pooled-vector plots show invariance. A labeled point/reflection figure supports the existing distance-versus-handedness test. Every new plot distinguishes architecture diagnostics from property prediction or learned importance. |

The changes add **nine new figures** and expand one existing figure. All are static, generated offline from visible calculations or tracked data, and accompanied by interpretation and guided questions. They use the existing dependencies; no graph framework, GPU, download, or added training sweep is required.

## Scientific distinctions retained

- Parts 1 and 2 retain explicitly synthetic, dimensionless neural-network training data. The new alcohol-mass case uses actual RDKit-calculated average molar masses in g/mol. It is not described as measured mass or solubility. The family rule is $\mathrm C_n\mathrm H_{2n+2}\mathrm O$; its known composition, rather than generic machine-learning extrapolation, supports the exact held-out calculation. The computed $\mathrm{CH_2}$ increment is about 14.027 g/mol under this atomic-weight convention.
- The loss-contour example fixes two of the original four parameters and updates only the displayed weight and bias. Its zero-loss line is a statement about the one-row fit, not parameter identifiability or predictive accuracy on new inputs.
- The hand-set ReLU construction demonstrates representational capacity without claiming training or chemistry. The wider-range MLP plot reuses the already selected model and a known synthetic generating function. No retuning follows that diagnostic.
- The solubility experiment preserves the exact local data checksum, 600-row cap before descriptors, whole-scaffold split, canonical-identity separation, training-only input/target scaling, validation checkpoint selection, and one frozen test comparison. The new split figure adds no test-target-driven choice.
- The reaction heatmap reads unchanged measured yields from the audited 3,955-row high-throughput screen. It selects the first sorted canonical ligand and base without looking at yield, keeps those conditions fixed, and displays individual observations rather than mixing conditions through averaging. It adds no reaction-yield predictor. A high observed yield motivates confirmation, not a general superiority or mechanistic claim. Missing combinations are distinct from zero yield; absent experimental context is not invented. See [dataset provenance](../datasets/README.md).
- Graph figures show possible information flow and computed numerical embeddings. They do not label arbitrary hidden values as atom importance, solubility, energy, or probabilities. Atom permutations and batch isolation remain tested, and omitted stereo information still produces the demonstrated collision.

## Validation

All **52 code cells** passed individual fresh kernels with a **10-second per-cell timeout**. Validation used the native Windows course environment with Python 3.12.14, PyTorch 2.11.0+cpu, scikit-learn 1.8.0, RDKit 2026.03.6, NumPy 2.5.3, and Matplotlib 3.11.2. The native environment paths and `MKL_THREADING_LAYER=SEQUENTIAL` were set before process startup; computation was limited to one CPU thread as in the notebooks.

| Notebook | Code cells | Sum of code-cell times | Slowest cell | Fresh-kernel wall time |
| --- | ---: | ---: | ---: | ---: |
| Chapter 11 Part 1 | 12 | 3.242 s | 2.426 s | 4.493 s |
| Chapter 11 Part 2 | 13 | 3.128 s | 2.091 s | 4.341 s |
| Chapter 11 Part 3 | 16 | 4.168 s | 2.896 s | 5.520 s |
| Chapter 11 Part 4 | 11 | 1.957 s | 1.215 s | 3.087 s |
| **Total** | **52** | **12.495 s** | **2.896 s** | **17.441 s** |

The checks retain the existing gradient/autograd agreement, held-out protocols, checkpoint round trips, loss and shape assertions, dataset audits, and graph symmetry tests. New checks independently verify the loss-slice update, atomic-weight formula, held-out molar masses, ReLU sum identity, split counts, nonduplicated reaction-panel cells, graph distances, permutation correspondence, and atom-sum readout. All nine new figures and the expanded activation figure were rendered and visually inspected. The gradient and ReLU derivative displays were refined, then their two notebooks rerun successfully.

Author-validation copies, extracted PNG figures, per-notebook timing records, and a combined report are in the ignored `outputs/full_review_chapter11/validation/` directory. All four source notebooks have cleared outputs for the parent agent's final course-wide run. All notebook markdown uses `$...$` or `$$...$$` for equations. No shared dependency files, datasets, README, validator, or historical review records were changed.

## Primary sources checked

- [PyTorch 2.11 Linear](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Linear.html) and [autograd mechanics](https://docs.pytorch.org/docs/2.11/notes/autograd.html): affine maps, differentiation, and nondifferentiable-operation conventions.
- [RDKit descriptor API](https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html): calculated molecular-mass conventions and molecular descriptors.
- [Ahneman et al., reaction performance study](https://doi.org/10.1126/science.aar5169), [the authors' data/code repository](https://github.com/doylelab/rxnpredict), and the [course's exact provenance audit](../datasets/README.md): the measured reaction screen and its role definitions. The original publisher URL could not be fetched during this pass; the authors' repository and local matched-data audit were available.
- [Gilmer et al., Neural Message Passing for Quantum Chemistry](https://proceedings.mlr.press/v70/gilmer17a.html): shared graph messages, aggregation, and molecular readout.

[Course contents](../Readme.md) · [Historical Chapters 10–11 corrections](chapters10-11-review.md)
