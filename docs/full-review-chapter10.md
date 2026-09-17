# Beginner-focused review of Chapter 10

Both notebooks were read in full and improved for graduate students with no prior machine-learning background. Existing dataset provenance, curation, row alignment, grouped validation, fixed model candidates, baselines, and final assessment remain intact. No extra model fits, hyperparameter candidates, dataset downloads, or dependencies were added.

## Changes by notebook

### [Part 1: molecular features](../Chapter10_Part1.ipynb)

- Added a structure–descriptor–fingerprint–target–prediction bridge, defining feature rows/columns and the symbols used in supervised learning.
- Added a worked analogue-inspection case around phenol using the same twelve embedded structures, which have no measured labels.
- A two-panel figure combines a named mass/TPSA map with the existing Morgan-Tanimoto ranking. It preserves physical units, avoids a fitted embedding or invented activity score, and shows why two descriptors cannot distinguish the lactic-acid stereoisomers.
- A predeclared three-molecule inspection budget returns aniline, 4-nitrophenol, and benzene under the specified fingerprint. The text treats this as a structural inspection list, not a biological ranking or a prediction of experimental suitability.
- Added an explicit conclusion, next research step, and self-check about keeping representation selection separate from test-label inspection.

### [Part 2: evaluation](../Chapter10_Part2.ipynb)

- Defined fitting, loss, hyperparameters, validation, test data, baselines, and generalization before introducing the models.
- Added a diagram of the **actual** log D outer split and three inner folds. Each molecule's allowed role is visible, and assertions verify that the reserved test remains untouched and every training molecule validates exactly one inner fold.
- Added a retrospective screening interpretation of the **existing frozen BBBP predictions**. A 20% inspection budget is declared before fitting and recorded in the manifest. Rankings use source row to break exact ties, never test labels.
- The new cumulative-recovery and class-count figure compares recovered known positives with uniform random selection's expected count in the same finite pool. It distinguishes precision, positive recovery, prevalence, fixed-threshold classification, and calibration.
- The reviewed run inspects 33 of 163 held-out records and recovers 27 positives; uniform random selection from that pool would have expected 25.10. This descriptive difference is not a significance test, a tuned budget, or evidence of clinical utility. The source contains no new model selection based on it.
- Kept existing warnings about curated benchmark scope and future chemical distribution shift. Added a next-step research plan and a self-check on why adapting the budget to this test curve would require another independent assessment.

All Markdown math uses `$...$` and `$$...$$`. Existing exercises and answers remain, with additional interpretation prompts alongside the new cases. Source notebooks are handed back without outputs for final course-wide execution.

## References and scope

- [RDKit descriptor API](https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html): named structure-derived descriptors and their implementation. Original fingerprint, TPSA, and Wildman–Crippen references remain beside their definitions in Part 1.
- [scikit-learn precision/recall documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html): the distinction between positives among selected records and the fraction of all positives recovered. The finite-pool expected random count is derived directly as $kN_+/n$.
- [Local dataset provenance](../datasets/README.md): unchanged source identities, measured log D at pH 7.4, BBBP label meaning, and curation limitations. No new experimental labels were introduced.

Primary documentation checked on 16 September 2026. These figures organize existing structures and interpret frozen predictions; they do not add a causal, clinical, or prospective screening claim.

## Validation

Both notebooks passed in separate fresh kernels with a **10-second per-cell timeout**. The private report and executed copies are under ignored `outputs/full_review_chapter10_validation/`. All three new figures were inspected, including label placement, class counts, axes, and the reserved-test boundary. A long screening-figure title was shortened after the first inspection.

| Notebook | Code cells | Whole-notebook time | Slowest cell |
|---|---:|---:|---:|
| Chapter10_Part1 | 13 | 2.66 s | 0.886 s |
| Chapter10_Part2 | 17 | 5.27 s | 1.229 s |

The author run used Python 3.12, RDKit 2026.03.6, scikit-learn 1.8.0, NumPy 2.5.3, pandas 2.3.3, and Matplotlib 3.11.2 with one CPU thread and the course runtime environment. Timings depend on hardware and include rendered outputs. No shared validator, dependency files, datasets, or README were edited.
