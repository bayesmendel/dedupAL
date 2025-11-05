import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from src import config


def random_forest_training(X, y, verbose = True):
    """
    Trains a Random Forest classifier using pre-defined hyperparameters 
    for Steps 3 and 4 of the method defined in Section 2 of Rosito 
    et al. (2025)

    The model is fitted on the current set of labeled data, and the 
    confusion matrix is calculated on the training set.

    Parameters
    ----------
    X : pd.DataFrame
        Training features, consisting of absolute differences in 
        pedigree features (see Appendix A in Rosito et al. 2025).
    y : np.ndarray
        Training labels, where 0 is candidate negative (CN) and 1 is 
        candidate positive (CP), as defined in Rosito et al (2025).
    verbose : bool, optional
        If True, prints the training confusion matrix to the console (
        Default is True).

    Returns
    -------
    RandomForestClassifier
       The fitted Random Forest model.
    np.ndarray
       The training confusion matrix.
    """

    seed = config.seed
    class_weight = config.class_weight
    min_samples_split = config.min_samples_split
    max_depth = config.max_depth
    min_samples_leaf = config.min_samples_leaf
    model = RandomForestClassifier(random_state = seed, 
    		class_weight = class_weight, 
        	min_samples_split = min_samples_split,  
        	max_depth = max_depth, 
        	min_samples_leaf = min_samples_leaf)
    rf = model.fit(X, y)
    y_pred = rf.predict(X)
    cm = confusion_matrix(y, y_pred)

    if verbose:
    	print("Training confusion matrix")
    	print(cm)

    return rf, cm


def new_labeling(rf, X, verbose = True):
    """
    Determines new labels for the unlabeled candidate pairs based on 
    the predicted probabilities of the current Random Forest model.
    These labels are the core of Step 4 described in Section 2 of
    Rosito et al. (2025). 

    Pairs are assigned a label of 1 (candidate positive, CP) or 0 
    (candidate negative, CN) if thier predicted probabilities exceed 
    the global parameter p_threshold, or 3 if they remain unlabeled

    Parameters
    ----------
    rf : RandomForestClassifier
        The currently fitted Random Forest model.
    X : pd.DataFrame
        Features of the unlabeled candidate pairs.
    verbose : bool, optional
        If True, prints the count of newly labeled positive, negative, 
        and remaining unknown pairs (Default is True).

    Returns
    -------
    np.ndarray
        A vector of new labels corresponding to the rows in X.
    """

    p_threshold = config.p_threshold
    p_pred = rf.predict_proba(X)
    n = len(X)
    y_new = 3 * np.ones(n)

    for i in range(n):
        if p_pred[i, 1] >= p_threshold:
            y_new[i] = 1
        elif p_pred[i, 0] >= p_threshold:
            y_new[i] = 0

    if verbose:
        print('Number of new CP: ', sum(y_new == 1))
        print('Number of new CN: ', sum(y_new == 0))
        print('Number unknown pairs: ', sum(y_new == 3))


    return y_new
  