# dedupAL: Interpretable Active Learning for Pedigree Data Deduplication in Cancer Genetics

## Overview
`dedupAL` is a Python package that implements an interpretable, Active Learning (AL)-based pipeline for identifying duplicate pedigree records in cancer genetics datasets. The goal is to efficiently label candidate pairs as either candidate positive (CP, likely duplicate) or candidate negative (CN, probably not a duplicate), significantly reducing the need for costly manual review.

The pipeline integrates three core components: heuristic rules, graph-based feature extraction, and an iterative Active Learning classification loop.

## Methodology

The deduplication pipeline operates on **pairs of pedigrees**, classifying each pair as either **Candidate Negative (CN, class 0)** or **Candidate Positive (CP, class 1)**. The overall process follows four main steps:

* **Steps 1 & 2: Heuristic-based Initial Labeling.** A subset of pairs receives initial labels using genetic mutation variant sharing and coincidence count heuristics.
* **Step 3: Initial Random Forest Training.** A Random Forest classifier is trained on this initial labeled dataset in which the features are based on pedigree graph representations. This model is used to predict confidence scores for the unlabeled pool.
* **Step 4: Iterative Active Learning.** The model training and confidence-based label prediction are repeated iteratively. This systematically refines the model and progressively labels the most certain remaining pairs.

   <img width="603" height="292" alt="al 2" src="https://github.com/user-attachments/assets/a24ef021-88f2-4669-acb1-fba7a461ff27" />

## Installation 

This package requires the installation of the following Python libraries:

* numpy & pandas: Core libraries for efficient data manipulation and numerical operations.
* scikit-learn: Provides the Random Forest classifier and model training utilities.
* networkx: Used for building and analyzing pedigree graphs for feature extraction.

`pip install numpy pandas scikit-learn networkx`

## Configuration

All critical thresholds and constants are managed in `src/config.py`. These values are directly referenced by the dedupAL function:

| Parameter | Description | Reference Section |
| :--- | :--- | :--- |
| `N_threshold` | Coincidence count threshold for initial positive labeling (Step 2). | Sec. 2 B.2 |
| `p_threshold` | Probability threshold for confident Random Forest predictions (Steps 3 and 4). | Sec. 2 B.3 |
| `T` | Maximum number of Active Learning iterations. | Sec. 2 B.4 |

The `src/config.py` file also sets the Random Forest hyperparameters and the list of cancers used for feature extraction. Consequently, the input dataset must contain columns matching the names of those cancers.

## Dataset

The input dataset must be a table where each row represents a single individual. Families must have a unique identifier (default: Pedigreename), which is the column used for pairing. This table must include the following columns:

* `UPN`: Identifier for each individual.
* `Mother.ID` and `Father.ID`: Identifiers for parental relationships.
* `Sex`: Coded sex (0 for female, 1 for male).
* `YearOfBirth`: The individual's year of birth.
* `isDead`: Deceased status (1 if dead, 0 otherwise).
* `HadCancer`: General cancer status (1 if the individual had any cancer, 0 otherwise).
* `[Cancer Name]`: Binary column (1/0) for each specific cancer type defined in `src/config.py`.
* `Variant`: The specific genetic variant (mutation) identified in the individual.
* `Presult`: Status indicating a positive genetic test result (1/0).

## Usage Example

The `dedupAL` function manages the entire pipeline.

```
from src.dedupAL import dedupAL

# Load your dataset
data_path = 'path/to/your/combined_data.csv' 
df = pd.read_csv(data_path)

# Define your reference year
current_year = 2025 

# Run the deduplication workflow
# The function returns the final counts, the global indices of the pairs,
# and the number of unlabeled pairs per iteration.
# results = [CP_count, CN_count, Unlabeled_count]
results, index_pos, index_neg, index_nolabel, remaining = dedupAL(
    dataframe=df, 
    current_year=CURRENT_YEAR,
    colname='Pedigreename', # Customize if your ID column is different
    verbose=True
)

print("\n--- Final Results Summary ---")
print(f"Candidate Positive Pairs: {results[0]} ({len(index_pos)} indices)")
print(f"Candidate Negative Pairs: {results[1]} ({len(index_neg)} indices)")
print(f"Remaining Unlabeled Pairs: {results[2]} ({len(index_nolabel)} indices)")
```
Plot of the number of unlabeled pairs as a function of the iteration:

<img width="565" height="414" alt="n-iter" src="https://github.com/user-attachments/assets/c0b045be-d763-482c-8da0-b33ccf73c0b4" />

See `docs/README.md` for supplementary presentation materials (slides/posters) related to this project.
