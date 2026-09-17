# Chapters 10–11: molecular machine learning and neural networks

> Historical correction record. The later [full-course teaching review](full-course-review.md) supersedes the notebook counts, dependency scope, and runtime totals below.


## Scope

All five original notebooks in Chapters 10–11 have been revised, and [Chapter 11 Part 4](../Chapter11_Part4.ipynb) adds a short graph-neural-network lesson. The six notebooks run independently, use embedded examples or tracked course CSV files, and require no network access after setup. Learning objectives, exercises, answers, primary references, and verified outputs accompany the lessons.

| Notebook | Revised focus | Main checks and improvements |
| --- | --- | --- |
| [10.1: Molecular features](../Chapter10_Part1.ipynb) | Twelve illustrative molecules, named descriptors, fingerprint generators, and similarity | Descriptor units and definitions; Morgan radius/chirality; bit/count differences; collisions; MACCS indexing; atom-order invariance; protonation and stereochemical information loss |
| [10.2: Evaluation and classical models](../Chapter10_Part2.ipynb) | Small log D regression and BBBP classification studies | Explicit source/structure audits; preserved row identities; scaffold-separated holdouts; training-only preprocessing and cross-validation; bounded model selection; dummy baselines; appropriate metrics and error diagnostics |
| [11.1: Perceptrons and gradients](../Chapter11_Part1.ipynb) | AND perceptron, single-point gradient calculation, and synthetic linear regression | Hard-threshold perceptron versus affine/logistic units; analytic/finite-difference/autograd agreement; parameter identifiability; tensor shapes; SGD versus ordinary least squares; complete inference checkpoint |
| [11.2: Multilayer perceptrons](../Chapter11_Part2.ipynb) | Small nonlinear synthetic regression | Why activations are needed; training-only feature/target scaling; validation patience; copied best checkpoint; held-out assessment; bounded CPU training; multioutput shapes; architecture/preprocessing/weights round trip |
| [11.3: Chemical applications](../Chapter11_Part3.ipynb) | Measured solubility regression, BBBP label/loss audit, and reaction-yield data audit | Grouped chemical splits; restored validation model; fixed mean/ridge baselines; correct log S units; complete preprocessing persistence; logits versus probabilities; role-preserving reaction features and different component-holdout questions |
| [11.4: Graph neural networks — new](../Chapter11_Part4.ipynb) | Small manual PyTorch message-passing network | Directed edges, atom/bond features, shared updates, invariant pooling, equivariant atom outputs, disconnected batching, a differentiability check, omitted chirality, and the distinction between connectivity and geometry |

## Scientific and methodological corrections

### Molecular data and evaluation

The original examples fitted scaling, variance filters, or PCA before splitting data. The revised supervised workflows fit data-dependent preprocessing only on the appropriate training partition, including inside cross-validation. Model choices use training/validation information; the final held-out comparison is explicitly separated from those choices.

Identical molecules and selected chemical groups are kept together. The chapters explain that scaffold splits answer a particular transfer question and do not guarantee deployment validity. They report the large shared acyclic group and unequal split sizes where relevant. A fixed structural descriptor can be computed independently for each molecule; learned preprocessing has a different information dependency.

The exhaustive sequence of many estimators is replaced by small, predeclared comparisons. Chapter 10.2 uses two ridge penalties and a 32-tree random forest for regression, and two logistic-regression penalties for classification. Training folds select the model. Mean/median/prior baselines, residuals, class counts, balanced accuracy, ranking metrics, and probability scores make limitations visible. R² can be negative; classification accuracy alone can conceal poor minority-class performance. Average precision is not treated as identical to a trapezoidal precision–recall area or as a calibrated clinical probability.

The revised notebooks distinguish experimental log D at pH 7.4 from RDKit's calculated log P, and measured log S from upstream ESOL predictions. Descriptor and fingerprint feature definitions are saved explicitly. Hash collisions, missing stereo, protonation choices, and missing assay conditions are treated as limitations of the representation, not information a model can recover automatically.

### Neural-network concepts and training

The original use of “perceptron” for unrestricted linear regression is replaced by a precise comparison of hard-threshold classification and differentiable affine/logistic units. A single labeled observation cannot identify four independent affine parameters; the notebook now demonstrates that rank limitation and verifies the actual gradient numerically and with autograd.

Repeated 5,000–10,000-epoch demonstrations and larger 1,000-epoch molecular networks are replaced by small CPU calculations. The nonlinear synthetic MLP has a 500-epoch maximum and validation patience; it stopped after 128 epochs and restored epoch 98 in the final run. The molecular MLP has a 120-epoch maximum; it stopped after 50 epochs and restored epoch 35. These short runs teach optimization and evaluation without implying that a low training loss proves chemical accuracy.

Checkpoints preserve architecture, preprocessing statistics, feature order, and model state, and their reconstructed predictions are checked. Target scaling is reversed before reporting metrics in physical target units. Regression, binary logits, and multioutput tensor shapes are handled explicitly. Locally generated checkpoint files are loaded with `weights_only=True`; the course does not download executable model artifacts.

The chemical applications notebook completes one measured-data neural regression workflow. BBBP and reaction yield remain substantial audit and interpretation sections rather than repeating expensive training loops. The reaction lesson compares familiar-component combinations with held-out aryl halides and keeps component roles distinct in concatenated fingerprints. It does not invent a reaction-yield model score.

### Modern graph representations

The new graph lesson implements message passing directly with PyTorch. It verifies node equivariance, molecular-output invariance, and separate-versus-batched computation. Its single oxygen-count gradient step is an architecture diagnostic with an exact counting baseline, not a molecular-property benchmark. The enantiomer example demonstrates an information collision caused by omitted stereochemistry. A separate point-geometry example shows why distances alone do not distinguish mirror reflection. The final section relates descriptors, graphs, SMILES/attention models, and 3D features without declaring a universal best architecture.

## Dataset provenance

The four chemistry CSV files were preserved. The new [dataset notes](../datasets/README.md) and [machine-readable provenance](../datasets/provenance.json) record hashes, target definitions, and independent comparisons with public snapshots:

- All 1,121 local solubility records match the measured column of DeepChem's 1,128-row Delaney table within numerical serialization precision. Seven upstream rows are absent; the original course export history remains unknown.
- All 4,200 lipophilicity SMILES/target pairs match DeepChem's experimental log D table.
- BBBP is byte-identical to the inspected 2,050-row source file. Current parsing identifies 11 invalid rows, 64 extra canonical-duplicate rows, and 10 canonical groups with conflicting labels.
- All 3,955 reaction rows, component strings, order, and yields exactly match the inspected `FullCV_01` worksheet, with the target column renamed. The table contains 44 distinct component strings.

Different labels for the same stored structure can reflect missing conditions or repeated measurements, so they are not automatically declared experimental errors. Each notebook states its own curation and grouping policy and saves the selected original row IDs. The classroom subsets are not official MoleculeNet benchmark splits.

## Validation and runtime

On 2026-09-15, all **70 code cells across six notebooks** passed serial execution in separate fresh kernels with a **10-second timeout per cell**. Source notebooks contain the verified outputs. Total notebook time, including kernel startup and shutdown, was **26.40 seconds**; the slowest individual cell took **2.849 seconds**.

| Notebook | Executed code cells | Notebook time | Slowest cell | Result |
| --- | ---: | ---: | ---: | --- |
| 10.1 | 12 | 2.31 s | 0.885 s | Passed |
| 10.2 | 15 | 6.25 s | 1.221 s | Passed |
| 11.1 | 10 | 4.62 s | 2.143 s | Passed |
| 11.2 | 11 | 4.69 s | 2.110 s | Passed |
| 11.3 | 14 | 6.19 s | 2.849 s | Passed |
| 11.4 | 8 | 2.34 s | 1.165 s | Passed |

All six notebooks received independent scientific/code review. Checks also covered notebook schema, code syntax, sequential execution, absence of error outputs, local navigation, checkpoint reconstruction, and 13 rendered figures. `pip check` reported no broken requirements.

```sh
python scripts/validate_chapters.py --chapters10-11 --cell-timeout 10 --inplace
```

The [validator](../scripts/validate_chapters.py) also accepts `--chapter10` and `--chapter11`. With no selector it covers all 27 revised course notebooks. Its version and per-cell timing report is written to `outputs/validation/report.json`; the final six-notebook record is retained locally as `outputs/validation/report-chapters10-11.json`. The timeout bounds validation rather than guaranteeing identical runtimes on every computer.

### Software and the Windows numerical-library fix

The tested environment uses Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, pandas 2.3.3, Matplotlib 3.11.2, RDKit 2026.3.6, scikit-learn 1.8.0, and PyTorch 2.11.0+cpu. The new requirements include scikit-learn and CPU PyTorch; no GPU runtime, TensorBoard, or graph framework is needed.

Actual neural-layer execution exposed an OpenMP runtime conflict between the Windows Conda/MKL numerical stack and the PyTorch wheel. The [environment file](../environment.yml) now sets `MKL_THREADING_LAYER=SEQUENTIAL`, an [Intel-supported backend selection](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2024-0/dynamic-select-the-interface-and-threading-layer.html), before Python starts. The validator applies the same setting along with one-thread limits. Reactivate an updated Conda environment before launching Jupyter so this setting takes effect.

After that change, **24 additional code cells** in Chapter 8 Part 5 and Chapter 9 Part 2 passed fresh-kernel regression checks, covering actual optimization, gradients, frequencies, thermochemistry, transition-state verification, and IRC endpoint refinement. Their original source outputs were preserved. This was a targeted compatibility check; Chapters 1–9 were not all reexecuted during this final-chapters review. Earlier validation records remain in their respective review notes.

## Interpretation and remaining limits

- The small chemical models illustrate a reproducible experiment. Their selected-subset scores do not establish state-of-the-art performance, broad applicability, or a robust ranking of algorithms. In particular, the BBBP classifier has poor specificity despite apparently respectable overall accuracy and average precision.
- Validation and test sets contain limited numbers of chemical groups, not independent observations of every possible scaffold. A single split and seed are not a complete uncertainty analysis. Missing assay conditions and selection effects remain important.
- Synthetic-function demonstrations validate mathematics and code. Molecular graph invariance validates a representation property. Neither establishes experimental accuracy.
- The runtime limits deliberately exclude broad hyperparameter searches, large Gaussian-process fits, large neural models, and pretrained-model downloads. More expensive research protocols are discussed without adding long optional execution cells.
- All eleven course chapters have now been reviewed. Legacy helper modules and files not used by the revised lessons have not received a separate general-purpose maintenance audit.

## Principal references

References are placed beside the relevant notebook explanations. Core sources include [RDKit fingerprint generators](https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html), [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [MoleculeNet](https://doi.org/10.1039/C7SC02664A), [PyTorch autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html), [saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [Gilmer et al.'s message-passing framework](https://proceedings.mlr.press/v70/gilmer17a.html), and [DimeNet](https://arxiv.org/abs/2003.03123). The [dataset notes](../datasets/README.md) link the original chemical-data publications and inspected snapshots.

[Course contents and setup](../Readme.md)
