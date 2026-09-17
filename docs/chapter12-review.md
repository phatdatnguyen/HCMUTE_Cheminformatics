# Chapter 12: graph neural networks

> Historical correction record. The later [full-course teaching review](full-course-review.md) supersedes the notebook counts, dependency scope, and runtime totals below.


## Scope and teaching sequence

Added on 2026-09-16. Chapter 11 Part 4 remains a short introduction and now links to this five-notebook sequence. All parts run independently from the repository root, offline after environment setup. The chapter uses existing RDKit, PyTorch, NumPy, pandas, Matplotlib and scikit-learn packages; no additional graph framework, GPU, large download or pretrained checkpoint is required.

| Notebook | Main lesson | Executable evidence |
| --- | --- | --- |
| [12.1 Molecular graphs](../Chapter12_Part1.ipynb) | Chemical input policy, categorical features, chirality, directed edges, graph metadata and disjoint batching | Invalid-input audit, unknown categories, empty edges, stereochemical distinctions, node/edge permutations, graph isolation and a failing whole-batch-centering example |
| [12.2 Message passing](../Chapter12_Part2.ipynb) | Bond-conditioned messages, neighbor aggregation, residual updates, graph readout, GCN/GIN/GAT/D-MPNN and a PyG bridge | Hand-checkable messages, autograd/finite differences, all-layer permutation tests, finite receptive field and directed-bond reverse exclusion |
| [12.3 Measured solubility](../Chapter12_Part3.ipynb) | A complete small graph-level prediction experiment with baselines | Source audit, fixed scaffold groups, training-only target scaling, validation checkpoint selection, untouched test comparison and checkpoint round-trip |
| [12.4 Diagnostics](../Chapter12_Part4.ipynb) | Expressivity limits, oversmoothing, attribution and explanation checks | Stationary-weighted mixing, an abstract regular-graph collision, input-gradient finite differences, integrated-gradient completeness, baseline dependence, parameter sensitivity and tensor-edit controls |
| [12.5 Geometric learning](../Chapter12_Part5.ipynb) | Radius graphs, smooth filters, invariance/equivariance, energies, forces, periodicity and modern atomistic models | Generated conformer, rotation/translation/permutation/reflection checks, net force/torque identities, finite differences and a cutoff scan |

Every notebook includes learning objectives, worked explanations, exercises with answers, primary references and saved provenance/diagnostics under the ignored `outputs/chapter12_part*/` directories. The notebooks intentionally repeat their small encoding/model definitions so learners can restart a lesson without running another notebook or loading a hidden helper module. Each part declares its feature schema: the feature columns are not interchangeable across lessons.

## Scientific and implementation choices

- **GNNs are an important model family, not an automatic performance winner.** The chapter explains their useful structure: variable-size inputs, shared neighborhood computations, and symmetry-aware geometric extensions. Descriptor, fingerprint and other representation approaches remain relevant comparisons. No architecture ranking is inferred from this short experiment.
- **Chemical state is part of the input policy.** Graph construction addresses explicit/implicit hydrogens, formal charge, radicals, isotopes, unknown categories, salts, stereo and unsupported bond types. Part 1 uses absolute CIP labels rather than treating order-dependent CW/CCW tags as absolute atom properties. Richer input still does not guarantee complete stereochemical information or accurate predictions.
- **Batching is a calculation, not a data split.** Node offsets and membership vectors route information between the correct atoms and molecular readouts. Tests include entirely edgeless batches and an intentionally corrupted cross-graph edge. Whole-batch centering demonstrates that disjoint edges alone do not guarantee independence from batch companions.
- **Equations agree with implemented operations.** Part 2 uses learned bond-type matrices and atom states; its directed-bond section separately explains persistent oriented-edge states and reverse-message exclusion. Part 3 uses a simpler bond-aware message function. Neither model is presented as a reproduction of a named research benchmark.
- **Learning and assessment are separated.** Part 3 fits target statistics and baselines on training rows, selects an actual copy of the best GNN weights on validation rows, and evaluates fixed recipes on test rows. The experiment saves identities, features, scales, versions, weights, history, predictions and protocol. It includes the correct masking of missing assay labels without treating them as negatives.
- **Symmetry and chemical accuracy are different claims.** Parts 2, 4 and 5 explicitly label controlled or untrained scores. Part 5's scalar output and force derivatives are dimensionless; no physical energy scale is assigned to random weights. Tests validate derivatives and transformations, not agreement with quantum mechanics.
- **3D assumptions are explicit.** The final notebook distinguishes SE(3) from E(3), polar-vector from pseudoscalar behavior, radius from covalent edges, and a generated conformer from an ensemble. Its cosine envelope is C1 but generally not C2; masking a neighbor list alone does not guarantee smooth forces or Hessians. Periodic edges require lattice-image shifts, and a local cutoff does not supply all long-range physics.

## Measured-property data and results

Part 12.3 reuses the unchanged [Solubility.csv](../datasets/Solubility.csv). The [dataset notes](../datasets/README.md) and [provenance manifest](../datasets/provenance.json) document its correspondence to the measured Delaney/ESOL source column. The target is `log10(S / (1 mol/L))`, not the ESOL model's predicted solubility. Missing experimental pH and temperature are not invented.

The source SHA-256 remains `3fedd5ad80f9231bd331929ba0943a117d0d6ee3f75eda9a27fab9b4ab974ad5`. All 1,121 rows parse; there are 11 repeated canonical-structure extra rows and six canonical groups with distinct labels. Repeats are retained within one partition. All source rows meet the declared connected, 1–60-heavy-atom eligibility policy; a label-independent SMILES hash selects 400 for bounded execution.

Exact non-stereochemical Murcko scaffold hashing yields **245 training, 19 validation and 136 test rows**. All acyclic molecules belong to one group; **111 test rows** are in that group. The small validation sample and uneven group sizes limit inference from aggregate metrics. Distinct scaffolds can still be similar, and the split is not an official MoleculeNet benchmark. The 400-row subset also differs from Chapter 11 Part 3's selection; compare models within this notebook.

The model has 4,921 parameters, hidden width 24, two message-passing rounds, mean atom pooling plus an explicit log atom-count feature, and a scalar regression head. Fixed one-hot elements and atom counts/flags yield 17 node features; seven edge features encode bond type, conjugation and ring membership. The selected data have no unknown elements, explicitly labeled isotopes, radical atoms or specified chiral atoms, but the encoder's omitted information still limits future inputs.

The fixed training recipe uses Adam with learning rate 0.01, maximum 80 full-batch epochs, patience 12 and tolerance 0.0001. The tested run reaches the 80-epoch cap and restores epoch 73. Baselines use the identical training rows: a training mean, nine-descriptor standardized ridge regression (`alpha=1`), and a 64-tree random forest on radius-2, 1,024-bit nonchiral Morgan fingerprints.

| Model | Test MAE (log units) | Test RMSE (log units) | Test R² |
| --- | ---: | ---: | ---: |
| Training mean | 1.863 | 2.196 | -0.569 |
| Descriptor ridge | 0.833 | 1.011 | 0.667 |
| Morgan forest | 1.575 | 1.857 | -0.123 |
| Small GNN | 0.924 | 1.236 | 0.503 |

The descriptor baseline outperforms this GNN in this experiment. Model settings were not changed to make the GNN win after examining test scores. The result is useful teaching evidence about concrete recipes on a difficult, small split; it is not a claim about the best achievable performance of each family. The notebook saves a cyclic/acyclic breakdown and training-similarity context, while explicitly avoiding claims that a Tanimoto score supplies calibrated uncertainty.

## Environment and validation

The new [requirements file](../requirements-chapters1-12.txt) includes the existing Chapter 11 requirements without adding packages. [environment.yml](../environment.yml) points to the new complete-course file and retains `MKL_THREADING_LAYER=SEQUENTIAL`, the supported threading choice used for the tested Windows Conda/PyTorch combination. An already functioning Chapter 11 environment can run this chapter.

[The validation script](../scripts/validate_chapters.py) now includes all 32 course notebooks by default and accepts `--chapter12` for these five. It executes fresh kernels, records package versions and per-cell timings, fails on errors and can save successful outputs into source notebooks:

```sh
python scripts/validate_chapters.py --chapter12 --cell-timeout 10 --inplace
```

All **65 code cells passed** fresh-kernel execution on 2026-09-16 with a 10-second cell timeout. Total notebook execution time was **24.94 seconds**, including kernel startup and rendering; the slowest cell was **4.75 seconds**. Verified outputs are saved in all five source notebooks, including 14 static PNG figures.

| Notebook | Code cells | Total seconds | Slowest cell, seconds |
| --- | ---: | ---: | ---: |
| 12.1 | 16 | 4.80 | 3.207 |
| 12.2 | 10 | 3.67 | 2.091 |
| 12.3 | 15 | 7.83 | 4.752 |
| 12.4 | 12 | 4.23 | 2.064 |
| 12.5 | 12 | 4.41 | 2.138 |

The longest cells loaded libraries; the GNN training loop itself took about half a second. First-time loading on this machine was slower than the final run (one initial combined-import cell took about 10.6 seconds). These timings describe the tested machine and software, not a guarantee for every computer. The examples bound molecular size, graph size, integration points, and training iterations to keep the actual calculations short.

The tested Windows environment used Python 3.12.14, PyTorch 2.11.0+cpu, RDKit 2026.03.6, NumPy 2.5.3, pandas 2.3.3, Matplotlib 3.11.2, scikit-learn 1.8.0, ipykernel 7.3.0 and nbclient 0.10.4. `pip check` passed, and the complete Chapter 12 requirements were satisfied without new installations. The generated [timing/version report](../outputs/validation/report-chapter12.json) is local and ignored by Git; the table above preserves its main results in the course documentation.

Independent source reviews covered graph/CIP policy, message-passing indexing and equations, learned preprocessing and checkpoint handling, and geometric symmetry/force mathematics. Generated figures were inspected for labels, scales and interpretation. Static checks cover notebook format, Python syntax, local links, executed outputs and validator selection.

Only Chapter 12 requires execution for this addition: earlier notebook code and installed package versions were not changed. Chapter 11 Part 4 receives navigation/conclusion prose only. Historical Chapters 1–11 validation records remain in their existing review notes; this work does not claim a new full-course execution.

## Sources and continued study

Sources appear next to relevant claims in the notebooks. Selected starting points:

- [Gilmer et al., Neural Message Passing for Quantum Chemistry](https://proceedings.mlr.press/v70/gilmer17a.html) and [Yang et al., molecular representation evaluation and D-MPNN](https://arxiv.org/abs/1904.01561).
- [Deng et al., systematic study of molecular property prediction](https://www.nature.com/articles/s41467-023-41948-6) for the importance of data, evaluation and baseline comparisons.
- [RDKit documentation](https://www.rdkit.org/docs/RDKit_Book.html), [PyTorch 2.11 indexed addition](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.index_add_.html), and [PyG message-passing guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html) for representations and implementation.
- [Integrated Gradients](https://proceedings.mlr.press/v70/sundararajan17a.html) and [saliency sanity checks](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html) for studying what an attribution actually measures.
- [SchNet](https://arxiv.org/abs/1706.08566), [DimeNet](https://arxiv.org/abs/2003.03123), [PaiNN](https://arxiv.org/abs/2102.03150), [EGNN](https://proceedings.mlr.press/v139/satorras21a.html), [NequIP](https://arxiv.org/abs/2101.03164), and [MACE](https://arxiv.org/abs/2206.07697) for geometric model families.

[Course contents](../Readme.md) · [Chapters 10–11 review](chapters10-11-review.md)
