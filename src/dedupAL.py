from src.active_learning import *
from src.graph_features import *
from src.build_pedigree_graphs import *
from src.coincidences import *
from src import config
from itertools import combinations


def dedupAL(dataframe, current_year, colname = 'Pedigreename', verbose = True):
    """
    Implements the full Active Learning (AL) based on the pedigree deduplication 
    pipeline described in Rosito et al. (2025).

    The function orchestrates all steps: initial variant-based labeling, graph
    feature extraction, initial Random Forest training, and the iterative AL loop.

    The output provides the global indices of the final candidate positive (CP),
    candidate negative (CN), and remaining unlabeled pairs, allowing for
    further manual review or analysis.

    Pipeline Steps:

    1. Step 1 setup: Generation of variant sets for families.
    2. Step 2 setup: Computation of coincidence counts between pairs.
    3. Steps 1 and 2 (initial labeling): Use of variant sharing and coincidence 
    counts to label a portion of pairs as CN (0) or CP (1) based on N_threshold.
    4. Steps 3 and 4 setup: Generation of pedigree graphs and extraction of features 
    comparing families.
    5. Step 3 (initial Random Forest): Training of the first Random Forest model on 
    the initially labeled data.
    6. Step 4 (active learning loop): Iteratively use of Random Forest models 
    to predict probabilities to label the most confident remaining unlabeled pairs, 
    retrain the model, and repeat until no more pairs can be labeled or the maximum
    number of iterations (T) is reached. 

    Parameters
    ----------
    dataframe : pd.DataFrame
        The full dataset.
    current_year : int
        The fixed reference year for feature matrix calculation.
    colname : str, optional
        The name of the column containing the pedigree ID. 
        (Default is 'Pedigreename').
    verbose : bool, optional
        If True, prints training confusion matrices and number of new labeled pairs.

    Returns
    -------
    list
        A list containing the final counts of [CP pairs, CN pairs, unlabeled pairs].
    np.ndarray
        Global indices of the final Candidate Positive (CP) pairs.
    np.ndarray
        Global indices of the final Candidate Negative (CN) pairs.
    np.ndarray
        Global indices of the final unlabeled pairs.
    """


    print("--- Starting Deduplication Pipeline ---")

    families = dataframe[colname].unique()   
    pairs = np.array(list(combinations(families, 2)))
    npairs = len(pairs)

    dataframe['Variant'] = dataframe['Variant'].fillna('Reported as NaN')
    print(f"Families: {len(families)}. Candidate pairs: {len(pairs)}.")
    
    # --- Step 1 setup: Variant sets generation
    fam_varset = OrderedDict()
    
    for f in families:
        variants = dataframe.loc[dataframe[colname] == f, 'Variant'].unique()
        fam_varset[f] = {x for x in variants if x != 'Reported as NaN'}

    # --- Step 2 setup: Computation of the number of coincidences
    coincidences, proportions = coincidences_pairs(pairs, dataframe, 
                                current_year, colname)

    # --- Steps 1 and 2: Initial labeling
    y = 3 * np.ones(npairs)
    N_threshold = config.N_threshold

    for (i,p) in enumerate(pairs):
        set1 = fam_varset[p[0]]
        set2 = fam_varset[p[1]]
        if y[i] != 1:
            empty = set1.isdisjoint(set2)
            nc = coincidences[i] >= N_threshold
            if empty:
                y[i] = 0
            elif nc:
                y[i] = 1

    initial_labeled = sum(y != 3)            

    print('Step 1 and Step 2: Initial labeling completed')
    print(f"Initial labeled: {initial_labeled} ({initial_labeled/npairs*100:.2f}%)")

    # --- Steps 3 and 4 setup: Graph building and feature calculation
    pedigree_graphs = build_pedigree_graphs(dataframe, current_year, colname)
    graph_pairs = [(pedigree_graphs[i], pedigree_graphs[j]) for [i,j] in pairs]
    X = pd.DataFrame([extract_features(G1, G2) for G1, G2 in graph_pairs]).reset_index(drop=True)
    
    # --- Step 3: Initial Random Forest
    X_train = X.iloc[y != 3, :]
    y_train = y[y != 3]
    X_nolabel = X.iloc[y == 3, :]

    index_pos = np.where(y == 1)[0]
    index_neg = np.where(y == 0)[0]
    index_nolabel = np.where(y == 3)[0]

    rf, cm =  random_forest_training(X_train, y_train, verbose)

    print("Step 3: Initial training completed (iteration 1)")

    # ---  Step 4: Active learning loop
    niter = config.T
    for i in range(2, niter + 1):

        # Get new labels for the current X_nolabel pool
        y_new_labels_for_current_X_nolabel = new_labeling(rf, X_nolabel, verbose)
        rem = sum(y_new_labels_for_current_X_nolabel == 3)

        # Update the training set    
        newly_labeled_mask = (y_new_labels_for_current_X_nolabel != 3)
        X_trainable_batch = X_nolabel.iloc[newly_labeled_mask, :]
        y_trainable_batch = y_new_labels_for_current_X_nolabel[newly_labeled_mask]
        X_train = pd.concat([X_train, X_trainable_batch])
        y_train = np.concatenate([y_train, y_trainable_batch])
        # Track CP, CN, and unlabeled indexes
        newly_positive_global_indices = index_nolabel[y_new_labels_for_current_X_nolabel == 1]
        newly_negative_global_indices = index_nolabel[y_new_labels_for_current_X_nolabel == 0]
        index_pos = np.concatenate([index_pos, newly_positive_global_indices])
        index_neg = np.concatenate([index_neg, newly_negative_global_indices])
        index_nolabel = index_nolabel[y_new_labels_for_current_X_nolabel == 3]

        if rem == 0: # no more unlabeled pairs
            print('End')
            break

        # Update the unlabeled set
        X_nolabel = X.iloc[index_nolabel, :]

        # Retrain the Random Forest model with the updated, expanded training data
        rf, cm =  random_forest_training(X_train, y_train, verbose)        
        print('Iteration ', i, ' completed')

    print("Step 4: Active learning scheme completed")

    results = [len(index_pos), len(index_neg), len(index_nolabel)]

    return results, index_pos, index_neg, index_nolabel