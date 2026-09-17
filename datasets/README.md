# Course datasets: provenance and interpretation

The CSV files in this folder predate the course revision. Their rows and labels were preserved. On 2026-09-15, the four chemistry datasets were compared with public source snapshots. [provenance.json](provenance.json) records local and upstream SHA-256 hashes, comparison methods, target definitions, and structural audit counts. The original course export procedures are undocumented; the comparisons below establish data correspondence without inventing that history.

Required notebook calculations read the tracked local CSV files and run offline. Source downloads used for this review are kept only in the ignored `outputs/dataset_audit/` folder.

## What the targets mean

| Local file | Rows | Target | Verified correspondence |
| --- | ---: | --- | --- |
| [Solubility.csv](Solubility.csv) | 1,121 | `solubility`: log₁₀(S / (1 mol/L)) | Every local SMILES/label pair matches the **measured** solubility column in the 1,128-row DeepChem Delaney snapshot, after trimming SMILES whitespace and allowing 10⁻¹⁰ numerical tolerance. Seven upstream rows are absent. |
| [Lipophilicity.csv](Lipophilicity.csv) | 4,200 | `lipophilicity`: experimental octanol/water log D at pH 7.4 | All local SMILES/label pairs match the DeepChem `smiles`/`exp` columns. |
| [BBBP.csv](BBBP.csv) | 2,050 | `p_np`: reported binary permeability label, 1 positive and 0 negative | The local file is byte-identical to the inspected DeepChem snapshot. |
| [BuchwaldHartwigReactionYield.csv](BuchwaldHartwigReactionYield.csv) | 3,955 | `Yield`: reported reaction yield in percent | All four component strings, row order, and numeric labels exactly match sheet `FullCV_01` of the inspected reaction-yield workbook, with `Output` renamed `Yield`. |

The solubility labels are not the upstream **ESOL-predicted** column. A one-unit error in log S corresponds to a tenfold concentration ratio, not one mol/L. The lipophilicity labels are log D at a specified pH, not RDKit's calculated neutral-species log P. The solubility export lacks per-record temperature, pH, and measurement details, so those conditions should not be invented.

BBBP is a heterogeneous benchmark for a reported label. A model probability is not a validated clinical probability or a complete pharmacokinetic prediction. Reaction-yield errors in the recorded 0–100 scale are **percentage points**. The four-role table belongs to a particular high-throughput screen; it does not represent all possible Buchwald–Hartwig reactions or document every experimental condition.

## Sources

- **Molecular datasets:** [MoleculeNet paper](https://doi.org/10.1039/C7SC02664A), [DeepChem dataset documentation](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html), and the dataset loader sources for [Delaney](https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/delaney_datasets.py), [lipophilicity](https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/lipo_datasets.py), and [BBBP](https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/bbbp_datasets.py).
- **Solubility origin:** [Delaney, “ESOL: Estimating Aqueous Solubility Directly from Molecular Structure”](https://doi.org/10.1021/ci034243x). The inspected tabular snapshot is [DeepChem's processed Delaney file](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv).
- **Other molecular snapshots:** [DeepChem lipophilicity file](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv) and [BBBP file](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv).
- **Reaction-yield experiment:** [Ahneman et al., *Science* 2018](https://doi.org/10.1126/science.aar5169), with the [Doyle laboratory data/code repository](https://github.com/doylelab/rxnpredict).
- **Inspected reaction table:** the [Dreher/Doyle workbook](https://github.com/rxn4chemistry/rxn_yields/blob/master/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx) distributed by the authors of [*Prediction of chemical reaction yields using deep learning*](https://doi.org/10.1088/2632-2153/abc81d). Their [data preparation documentation](https://rxn4chemistry.github.io/rxn_yields/data/) attributes the data and splits to Ahneman et al. and Sandfort et al. The original experiment uses a fixed 4-methylaniline coupling partner; it is not a varying column in the course CSV.

These links provide attribution and the original data context; this review does not assign a new license to pre-existing third-party data.

## Structure and duplication audit

The following counts use RDKit parsing followed by canonical **isomeric** SMILES on successfully parsed records. They do not include tautomer standardization, salt removal, protonation changes, or experimental-condition reconciliation. The software version and invalid source-row indices are recorded in the JSON manifest.

| Dataset | Unparseable records | Extra rows with an already-seen canonical structure | Canonical structure groups with distinct labels |
| --- | ---: | ---: | ---: |
| Solubility | 0 | 11 | 6 |
| Lipophilicity | 0 | 0 | 0 |
| BBBP | 11 | 64 | 10 |

Distinct labels for the same stored structure can reflect repeats, missing measurement conditions, rounding, or actual inconsistency. They are not automatically erroneous measurements. The revised notebooks state their teaching curation choices, report excluded records, retain original row identifiers, and keep identical structures or selected chemical groups out of both sides of a train/test split. These choices change the evaluation population; the resulting subsets are not official benchmark splits.

The reaction table has four ligands, 22 additives, three bases, and 15 aryl halides: 44 unique component strings in total, all parsed successfully in the audit. No four-component tuple is repeated. A random row split can nevertheless put the same components on both sides. Holding out an aryl halide or an additive asks a different generalization question from holding out a new combination of familiar components.

## Other files

`IrisFlower.csv` and `IrisFlower.xlsx` support earlier introductory data-handling examples. They were outside this chemistry-label provenance audit. The revised Chapters 10–11 use the four chemistry CSVs above or explicitly labeled embedded numerical examples.

[Course contents](../Readme.md) · [Chapters 10–11 review](../docs/chapters10-11-review.md)
