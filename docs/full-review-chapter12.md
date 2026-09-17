# Teaching review and PyTorch Geometric expansion: Chapter 12

Reviewed on 2026-09-16. Chapter 12 now has **eight standalone notebooks**. The original five have clearer beginner introductions, first/deeper-pass routes, and additional worked visual examples. Three new lessons implement the graph-library workflow with actual PyTorch Geometric. The [course guide](course-guide.md#chapter-12-foundations-and-real-pytorch-geometric) explains how to study the sequence.

## Changes to the original five parts

| Notebook | New teaching bridge | Added visual and application |
|---|---|---|
| [12.1](../Chapter12_Part1.ipynb) | Molecule rows versus atom rows, tensors, embeddings, implicit H, and graph targets. | Deliberately exchange C/O feature columns while preserving shape; an independent oxygen-count check reveals the corrupted schema. Count and fraction plots explain different pooling meanings. |
| [12.2](../Chapter12_Part2.ipynb) | Explain shared messages, hidden vectors, forward passes, training, and neighborhood notation in words. | A synchronous three-atom update is worked by hand and drawn for two rounds. Exact checks give `[1,1,2] → [2,4,3] → [6,9,7]`. The library bridge now leads to the executable PyG sequence. |
| [12.3](../Chapter12_Part3.ipynb) | Features/targets, experimental partitions versus mini-batches, and logarithmic concentration units. | Apply a declared 1 mmol/L scenario to frozen test predictions; compare decision counts and cumulative absolute log errors. This is explicitly post-assessment interpretation, with no threshold optimization or refitting. |
| [12.4](../Chapter12_Part4.ipynb) | Separate representation capacity, predictive validity, and explanation; define gradients, references and completeness. | Plot the actual score along two attribution paths. Different starting scores create different contrasts despite a common endpoint. Fractional element vectors are computational constructs, not synthesizable intermediates. |
| [12.5](../Chapter12_Part5.ipynb) | Explain invariant scalars and equivariant arrows through moving versus deforming a molecule. | Prescribe two butane torsions, showing identical connectivity/fingerprints and different terminal-carbon distances. This motivates geometry-dependent input for conformer-specific targets without calling the geometries optimized minima. |

All five retain the existing graph, gradient, permutation, isolation, finite-difference and symmetry checks. The additions reuse existing data or tiny embedded examples; they add no model-selection sweep. Five new figures were rendered, inspected, and adjusted for legend space, label width and 3D viewing angle.

## New Part 6: real PyG data, batching and layers

[Chapter12_Part6.ipynb](../Chapter12_Part6.ipynb) uses actual `Data`, `Batch`, `DataLoader`, `MessagePassing`, `GCNConv` and `GINEConv`.

- A connected heavy-atom representation produces 17 atom features and seven bond features, with each chemical bond stored in both directions. Feature order, hydrogen policy and omitted information are explicit. A one-node water graph exercises empty edges.
- Actual molecular masses are calculated only as graph-label routing examples, not presented as experimental targets or learned predictions.
- `Batch.to_data_list()` restores each graph and its label/source identity. A shuffled `DataLoader` retains that association.
- A custom bond-aware `MessagePassing` layer agrees with a manual `index_add` calculation in both forward outputs and input gradients.
- `GCNConv` agrees with explicit symmetric adjacency normalization including self-loops. `GINEConv` demonstrably consumes edge features; a controlled tensor edit changes only the intended graph's readout.
- Permutation and single/batched prediction checks test behavior beyond tensor shapes. Three generated figures show molecules, the actual batch and bond-input sensitivity.

These are fixed-weight architecture diagnostics. Their scores are not learned chemical properties.

## New Part 7: measured-solubility regression with PyG

[Chapter12_Part7.ipynb](../Chapter12_Part7.ipynb) uses the unchanged measured-solubility source and the **same 400 selected rows and grouped split as Part 3**: 245 training, 19 validation and 136 test rows. The architecture and optimizer differ, so the comparison does not isolate a causal effect of using a library.

The model uses two 24-channel residual `GINEConv` layers, mean pooling, a log node-count feature, and an unrestricted scalar head. It trains in mini-batches of 64 with at most 60 epochs and validation stopping. Target scaling is fitted on training rows only. The selected weight state is cloned, restored and verified; the matched baselines use exactly the same rows.

The saved experiment includes split identities, source checksum, full feature schema, target scale, architecture, optimizer/training settings, versions, metrics and predictions. A complete checkpoint passes a `weights_only=True` reload check. Learning curves, split composition, test parity and actual high-residual molecules are displayed. The failure inspection occurs after assessment and is not used to remove records or retune the model.

In the checked run, the selected checkpoint is epoch 59 of 60. Test RMSE is **1.089 log units for GINE**, **1.011 for descriptor ridge**, and **2.196 for the training mean**. These results support comparing simple baselines. The 19-row validation partition and the 111 acyclic test rows belonging to one group sharply limit any broad ranking claim.

## New Part 8: BBBP classification and actual GNNExplainer

[Chapter12_Part8.ipynb](../Chapter12_Part8.ipynb) audits the unchanged 2,050-row source, removes invalid/disconnected records and contradictory canonical-label groups, then deduplicates and selects a bounded 500-row sample. The group-disjoint split has **286 training, 109 validation and 105 test rows**. All source-row identities and inclusion policies are retained.

The compact GINE classifier trains with a training-only class weight and restores its best validation weighted-BCE checkpoint. A Morgan logistic model and training-prevalence baseline provide comparisons. Each learned model selects a screening threshold from a fixed candidate list using **validation balanced accuracy**; test labels are used only for the frozen assessment. Weighted-loss sigmoid scores are explicitly not assumed to be calibrated probabilities.

In the checked run, GINE restores epoch 2 after 10 epochs. Its validation-selected threshold is 0.5; Morgan logistic selects 0.8. Test balanced accuracy is **0.674 for GINE** and **0.793 for Morgan logistic**. GINE's Brier score is 0.239, worse than the constant-prior baseline's 0.182 and Morgan logistic's 0.122. A favorable-looking confusion matrix or explanation cannot replace this predictive evidence.

An actual PyG `Explainer` with `GNNExplainer(epochs=25)` explains the model's binary decision for a validation molecule selected by source identity and size, without examining confidence or label. Two seeds produce node-feature masks and directed-edge masks. The code verifies finite shapes, evaluates masked predictions, and confirms the classifier's weights and original prediction remain unchanged. Averaging both directed masks is explicitly a **display convention** for each chemical bond. PyG's internal binary explanation convention uses logit zero, independently of a downstream screening threshold.

The example's original sigmoid score is about 0.484 and masked scores are about 0.474. Mean absolute directed-mask difference between seeds is about 0.232. These short fits preserve a similar output while producing different masks; neither convergence nor a chemical mechanism is established. Five figures show class support, training, classification assessment and the explanation comparison.

## Software and source verification

The tested environment contains **PyTorch 2.11.0+cpu** and **torch-geometric 2.8.0.post1**. That PyG release is pinned in [requirements-chapters1-12.txt](../requirements-chapters1-12.txt). The basic installation supports these layers and explainer without compiled graph extensions or a GPU. The three new notebooks read local course data and perform no downloads at runtime. Windows uses the course's `MKL_THREADING_LAYER=SEQUENTIAL` setting before scientific imports.

Primary sources used include [PyG introduction](https://pytorch-geometric.readthedocs.io/en/2.8.0/get_started/introduction.html), [batching](https://pytorch-geometric.readthedocs.io/en/2.8.0/advanced/batching.html), [GINEConv](https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.nn.conv.GINEConv.html), [explanation API](https://pytorch-geometric.readthedocs.io/en/2.8.0/modules/explain.html), the [GNNExplainer paper](https://arxiv.org/abs/1903.03894), and [calibration study](https://proceedings.mlr.press/v70/guo17a.html). The actual installed APIs were exercised, not inferred from documentation alone. The notebooks retain primary references for message passing, molecular representation, attribution and geometric learning, with dataset provenance linked separately.

## Validation

All eight notebooks passed author fresh-kernel execution with a 10-second cell timeout. All 17 new figures were inspected, and independent read-only reviews found no remaining concrete scientific or code errors in the additions. The final integration run and per-notebook timings are recorded in the [full-course review](full-course-review.md#final-validation); those results supersede the earlier author-run timings. Markdown equations use only `$...$` and `$$...$$`.

The limits are deliberate: bounded CPU examples, small chemical subsets, a narrow feature schema, short training, no prospective external validation, and no claim that the best model on these particular splits is generally best. Parts 1–2 and 4–6 isolate mathematical/representation behavior; Parts 3, 7 and 8 connect graph learning to measured targets.
