# A guide to learning cheminformatics

This course is for graduate students who know some introductory chemistry but may be new to programming, molecular modeling, or machine learning. Start with a concrete question, work through a small example, and explain what the result supports. You do not need to memorize every API or derive every equation before using a notebook.

Use the [setup instructions and notebook index](../Readme.md). All 35 notebooks run independently from the repository root; each includes its own imports and inputs. Earlier chapters provide concepts, not hidden variables or output files needed by later chapters.

## How to study a notebook

1. **Read the question and learning objectives.** Identify the input, expected output, units, and model assumptions.
2. **Predict one result before running.** Sketch the shape of a curve, count the atoms, or work out one array shape.
3. **Restart the kernel and run from the top.** A kernel is the Python process holding your variables. Restarting removes values left over from previous experiments.
4. **Read the figure before inspecting every implementation detail.** State what each axis and color means. Separate a computed result from an experimental measurement.
5. **Change one small thing.** Explain the expected effect first, then compare with the output. Keep the original result for reference.
6. **Write a three-sentence conclusion:** what happened, why the evidence supports it, and what remains untested.

**First pass** sections establish the main ideas and interpretation. **Deeper pass** sections expose derivations, numerical checks, and implementation details. Both are part of the graduate course; a first reading can defer the latter. The default computations remain small. Larger research calculations are discussion or extension material, and optional browser viewers are disabled by default.

## The course in three connected blocks

| Block | Central question | What you should be able to produce |
|---|---|---|
| Chapters 1–3: representations and motion | How do structures and coordinates become reproducible calculations? | A checked molecular record, a geometry analysis, and an interpretation of a short trajectory. |
| Chapters 4–9: electronic structure and reactions | Which physical model connects a structure to a property or a reaction? | A controlled calculation with units, convergence checks, and a justified comparison. |
| Chapters 10–12: learning from data | How can measured examples support predictions for new molecules? | An audited dataset, an appropriate split, a baseline comparison, and a reproducible model. |

For a computational-chemistry project, emphasize Chapters 2–9. For a molecular-data project, first complete Chapters 1–2, then emphasize Chapters 10–12; return to Chapters 3–9 to understand the physical meaning and limitations of labels. Neither route replaces the full-course learning objectives.

## Chapter-by-chapter learning route

| Chapter | Refresh just before starting | First-pass question | A worked application and its boundary |
|---|---|---|---|
| [1: Python, tables, molecules](../Chapter01_Part1.ipynb) | Amount of substance, concentration, atoms and bonds. No Python assumed. | What does one variable, table row, or molecular record represent? | Plan a dilution, inspect measured solubility, and screen structures by a specified substructure. Formula equality does not establish molecular identity. |
| [2: geometry and mechanics](../Chapter02_Part1.ipynb) | Coordinates, angles, vectors, potential energy. | How do coordinates determine distances and how does a slope guide optimization? | Compare butane conformer starts and their MMFF energies. A local minimum is not proof of the global minimum or a solution population. |
| [3: dynamics](../Chapter03.ipynb) | Position, velocity, force, time, numerical stepping. | How do motion and sampling differ from minimizing energy? | Analyze energy exchange and correlation in short trajectories. More stored frames do not necessarily supply independent observations. |
| [4: quantum foundations](../Chapter04.ipynb) | Probability, integrals as areas, vectors as basis expansions. | What is a quantum state, and how does a finite basis approximate it? | Compare contracted and independent hydrogen basis functions. Variational improvement applies under stated Hamiltonian and subspace assumptions. |
| [5: Hartree–Fock](../Chapter05.ipynb) | Electrons, spin, orbitals, matrices, optimization. | What is being optimized in SCF, and what approximation remains after convergence? | Separate water basis-set effects from correlation effects; interpret a torsion scan. SCF convergence does not establish chemical accuracy. |
| [6: DFT](../Chapter06.ipynb) | Electron density versus orbital, energy derivatives. | What is approximated by a density functional and a numerical grid? | Inspect density slices and grid/basis sensitivity. A single agreement is not a universal functional ranking. |
| [7: semiempirical methods](../Chapter07.ipynb) | Approximation, parameter fitting, reference energies. | What makes a fast model's reported quantity comparable? | Evaluate PM7 preoptimization using a separate HF energy and gradient. MOPAC formation enthalpies and HF electronic energies have different definitions and reference zeros. |
| [8: molecular properties](../Chapter08_Part1.ipynb) | Charge/spin, derivatives, frequencies, photon energy. | Which calculation and observable answer the property question? | Fragment-energy cycles, isotope shifts, spectral broadening, and local potential approximations. Model properties and spectra are distinguished from measurements. |
| [9: reactions](../Chapter09_Part1.ipynb) | Stoichiometry, equilibrium, free energy, rates. | Is the question about favorability, a pathway, speed, or the reaction record? | Derive a condensation driving force, inspect a saddle, and choose a quench time in a declared kinetic model. Favorability is not speed; a mapped graph is not a mechanism. |
| [10: molecular machine learning](../Chapter10_Part1.ipynb) | Tables, distributions, similarity, averages. | What information enters a model and what kind of new chemistry does the split test? | Compare descriptor/fingerprint models on measured logD and BBBP labels. A test score applies to its particular labels and partition. |
| [11: neural networks](../Chapter11_Part1.ipynb) | Functions, slopes, matrix multiplication, training/validation/test. | How do gradients change parameters, and how do we detect useful learning? | Measured-property and reaction-yield learning with baselines and explicit representations. A flexible network does not remove dataset bias. |
| [12: graph neural networks](../Chapter12_Part1.ipynb) | Connectivity, array shapes, gradients, evaluation. | How do atoms exchange information and combine it into a prediction? | Measured solubility and BBBP in PyG, plus explanation and geometric-symmetry checks. Architectural tests do not substitute for chemical validation. |

### Chapter 8: one property question at a time

The seven parts are separate lessons: neutral properties → ions and radicals → orbitals → energy surfaces → vibrations → electronic excitations → short quantum dynamics. Read the property's definition before selecting a method. In particular, an orbital gap, an excitation energy, and a vibrational frequency describe different quantities.

### Chapter 12: foundations and real PyTorch Geometric

| Part | Main activity | Evidence to inspect |
|---|---|---|
| [12.1](../Chapter12_Part1.ipynb) | Encode molecular graphs and batch them manually. | Molecule pictures, shape checks, feature-order corruption, pooling and atom renumbering. |
| [12.2](../Chapter12_Part2.ipynb) | Trace and implement message passing. | A three-atom numerical example, graph isolation, gradients, and receptive fields. |
| [12.3](../Chapter12_Part3.ipynb) | Train a small PyTorch GNN on measured solubility. | Grouped split, validation checkpoint, matched baselines, multiplicative concentration errors. |
| [12.4](../Chapter12_Part4.ipynb) | Diagnose representations and explanations. | Controlled averaging/collision examples, baseline paths, completeness and parameter tests. |
| [12.5](../Chapter12_Part5.ipynb) | Connect 3D networks to physical symmetry. | Same-graph conformers, radius graphs, rotated forces, finite differences and smooth cutoffs. |
| [12.6](../Chapter12_Part6.ipynb) | Use actual PyG data structures and graph layers. | `Data`, `Batch`, `DataLoader`, custom `MessagePassing`, `GCNConv`, `GINEConv`; manual/library agreement. |
| [12.7](../Chapter12_Part7.ipynb) | Train a PyG GINE solubility regressor. | Mini-batches, restored best weights, descriptor baseline, failures and reloadable checkpoint. |
| [12.8](../Chapter12_Part8.ipynb) | Train a PyG BBBP classifier and run `GNNExplainer`. | Label audit, class weighting, validation-selected thresholds, frozen test scores, mask stability. |

For a first practical PyG session, review 12.1–12.2, then run 12.6–12.7. Read 12.3 alongside 12.7: they use the same selected solubility rows and split, while the architectures and optimization differ. Use 12.4 with 12.8 to discuss interpretation. Part 12.5 is the physical/geometric extension and connects directly to Chapters 2–9.

## Just-in-time notation and units

Math in the course uses `$...$` inline and `$$...$$` for displayed equations.

| Notation | Read it as | Small example |
|---|---|---|
| $x_i$ | Entry $i$ of an array or value for object $i$; the context identifies the object. | Three atomic charges form a length-three vector. |
| $\sum_i x_i$ | Add the selected entries. | Sum atom contributions to get a molecular quantity. |
| $\bar x=\frac{1}{n}\sum_i x_i$ | Arithmetic mean. | Averaging all atoms and averaging one molecule's atoms are different operations. |
| $\Delta E=E_B-E_A$ | Final minus initial energy. | A negative value means B is lower under the specified model. |
| $\frac{df}{dx}$ | Local slope: change in output per small input change. | A force component is the negative energy slope with respect to its coordinate. |
| $\nabla E$ | The collection of energy slopes along all coordinates. | For $N$ atoms in 3D, there are $3N$ coordinate components. |
| $\int_a^b f(x)\,dx$ | Accumulated area with sign. | A probability density must integrate to one over its domain. |
| $XW$ | Matrix multiplication: weighted combinations of feature columns. | `(atoms, features) @ (features, channels)` gives `(atoms, channels)`. |
| $\hat y$ | A prediction of the target $y$. | Residual $\hat y-y$ must retain the target's units. |
| $\log_{10}(S/S_0)$ | Base-ten logarithm of a dimensionless ratio. | With $S_0=1$ mol/L, $S=0.001$ mol/L gives $-3$. |

### Chemistry reminders

- **Connectivity** identifies bonded atoms. A 2D depiction is a drawing of that graph; its page distances are not measured bond lengths.
- **Conformers** differ in spatial arrangement while retaining the molecular connectivity and relevant stereochemical identity. **Configurational stereoisomers** require stereochemical information beyond a plain connectivity graph.
- **Formal charge** is an electron-bookkeeping convention. A model's **partial charge** is a partition of its charge distribution; different definitions can give different values.
- **Multiplicity** is $2S+1$, with total spin quantum number $S$. Charge and multiplicity are separate inputs to electronic-structure calculations.
- **Energy, enthalpy, and Gibbs energy** answer different questions. A geometry scan often compares model potential or electronic energies; equilibrium usually requires free energies and stated conditions.
- **Solubility, logP, and logD** are different targets. LogD depends on conditions such as pH and speciation. Read the [dataset definitions](../datasets/README.md) before modeling.

### Useful checks before trusting a number

- Is a distance in Å or bohr? Is an energy per molecule in hartree or per mole in kJ/mol?
- Is the temperature in kelvin? Is a logarithm natural or base ten?
- Did you compare identical composition, charge, spin and reference states, or construct a balanced reaction?
- Is a row a compound, an assay observation, a conformer, or a reaction? Can two rows be dependent?
- Was a parameter learned from the training partition, selected using validation, or fixed in advance?

## Short research projects using the bundled material

These projects build on the existing small calculations. Agree on a bounded protocol before running; a larger computation is not required for a good scientific report.

### A. Molecular identity and data quality

Use Chapters 1, 9.4 and 10. Choose a small set of supplied structures or local records. Define an identity policy for fragments, charge, stereochemistry and duplicate observations. Produce a molecule grid and an audit table. Explain which records can be compared and which need missing experimental information. Preserve the original source rows and report exclusions.

### B. A property calculation with two controlled approximations

Use Chapters 5–8. Keep the molecule, geometry, charge and multiplicity fixed while changing one basis, functional or numerical setting at a time. Choose one property and its units in advance. Show a comparison plot and distinguish numerical convergence from a change of physical model. Reuse the short water examples; do not interpret the most expensive calculation as experimental truth.

### C. Reaction favorability, rate and product collection

Use Chapters 9.1–9.3. State whether each input is measured thermochemistry, a calculated barrier, or a deliberately illustrative rate parameter. Plot a driving force or concentration trajectory. Explain why equilibrium, rate and isolated yield require different evidence. Include one limiting-case or conservation check.

### D. A measured-property prediction experiment

Use Chapters 10–12. Define the target, chemical domain, grouping, baseline and primary metric before fitting. Compare two small models on identical rows. Save the split identities, preprocessing and complete checkpoint. Show a learning curve, a held-out error plot and a few identified failures. If test results inspire a new recipe, describe it as a future experiment requiring new validation; do not relabel a reused test set as independent evidence.

### E. An explanation audit

Use 12.4 and 12.8. Define the scalar or class being explained and the reference or masking convention. Show the original prediction, masked prediction, and results for the two specified seeds. Separate prediction fidelity, stability, chemical validity and causality. Design a plausible follow-up measurement; do not claim a colored atom proves a mechanism.

## Suggested graduate assessment

| Criterion | Weight | What earns credit |
|---|---:|---|
| Question and scope | 20% | A specific target, appropriate units, declared conditions and assumptions. |
| Reproducible method | 25% | Clear inputs, bounded calculations, correct bookkeeping or data partitions, saved provenance. |
| Evidence and visualization | 25% | Readable figures with units, meaningful comparisons, and checks that could reveal a mistake. |
| Interpretation | 20% | Conclusions proportional to the evidence; explicit remaining uncertainty and a useful next experiment. |
| Communication | 10% | A concise explanation that another student can follow and reproduce. |

A model that loses to a baseline or a hypothesis contradicted by a controlled calculation can earn full credit. The goal is a reliable scientific conclusion, not a favorable-looking score.
