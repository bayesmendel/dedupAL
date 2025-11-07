# --- Active Learning and Deduplication Parameters (Rosito et al. 2025) ---

# Cancer names used in the DataFrame columns.
CANCER_NAMES = [
    'Adrenal', 'Brain', 'Breast', 'Leukemia', 'Lung', 'Osteosarcoma',
    'Phyllodes', 'Sarcoma', 'Soft tissue sarcoma', 'Colorectal', 'Pancreatic'
]

# N_threshold: Minimum number of required coincidences for initial positive label (Step 2).
N_threshold = 3
# p_threshold: Probability threshold for automatic labeling in Active Learning (Step 4).
p_threshold = 0.85
# T: Maximum number of iterations for the Active Learning loop.
T = 7


# --- Random Forest Model Hyperparameters (Steps 3 and 4) ---

# seed: Random state for reproducibility.
seed = 2005
# class_weight: Used to handle imbalance in the training data.
class_weight = 'balanced'
# min_samples_split: Minimum number of samples required to split an internal node.
min_samples_split = 15  
# max_depth: Maximum depth of the trees.
max_depth = 20
# min_samples_leaf: Minimum number of samples required to be at a leaf node.
min_samples_leaf = 6