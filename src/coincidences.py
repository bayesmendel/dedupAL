import pandas as pd
import numpy as np
from src import config

################################################################################
# Function  definitions
################################################################################



def compute_matrix(ped, dataframe, current_year, colname = 'Pedigreename'):
    """
    Constructs the Family Feature Matrix (M_i) for a single pedigree.

    This matrix represents the count of individuals within the pedigree 
    stratified by cancer type, sex, and age quartile, as defined in 
    Section 2 B.2 of Rosito et al. (2025). It is the core input for the
    coincidence-based labeling (Step 2).

    Parameters
    ----------
    ped : str
        The unique identifier of the pedigree (family) to process.
    dataframe : pd.DataFrame
        The full dataset.
    current_year : int
        The fixed reference year for calculating 'age_at_present'.
    colname : str, optional
        The name of the column containing the pedigree ID. 
        (Default is 'Pedigreename')

    Returns
    -------
    np.ndarray
        The Family Feature Matrix M_i of size (N_cancer * N_age_range) x N_sex
        11 * 4 * 2 = 88 elements.
    """

    cancer_names = config.CANCER_NAMES

    df_copy = dataframe.copy()
    age_at_present = current_year - df_copy['YearOfBirth']
    q1 = age_at_present.quantile(q = 0.25)
    q2 = age_at_present.quantile(q = 0.5)
    q3 = age_at_present.quantile(q = 0.75)
    df_copy['age_at_present'] = age_at_present

    final_matrix = np.empty((0, 2), dtype = int)
    family = df_copy.loc[df_copy[colname] == ped, ]

    for c in cancer_names:
        matrix = np.empty((4, 2), dtype = int)
        # Sex == 0/1:  0 = female, 1 = male
        cond = (family[c] == 1) & (family['Sex'] == 0) & (family['age_at_present'] < q1)
        matrix[0, 0] = sum(cond)
        cond = (family[c] == 1) & (family['Sex'] == 1) & (family['age_at_present'] < q1)
        matrix[0, 1] = sum(cond)
        
        cond = (family[c] == 1) & (family['Sex'] == 0) & (family['age_at_present'] >= q1) & (family['age_at_present'] < q2)
        matrix[1, 0] = sum(cond)
        cond = (family[c] == 1) & (family['Sex'] == 1) & (family['age_at_present'] >= q1) & (family['age_at_present'] < q2)
        matrix[1, 1] = sum(cond)

        cond = (family[c] == 1) & (family['Sex'] == 0) & (family['age_at_present'] >= q2) & (family['age_at_present'] < q3)
        matrix[2, 0] = sum(cond)
        cond = (family[c] == 1) & (family['Sex'] == 1) & (family['age_at_present'] >= q2) & (family['age_at_present'] < q3)
        matrix[2, 1] = sum(cond)

        cond = (family[c] == 1) & (family['Sex'] == 0) & (family['age_at_present'] >= q3)
        matrix[3, 0] = sum(cond)    
        cond = (family[c] == 1) & (family['Sex'] == 1) & (family['age_at_present'] >= q3)
        matrix[3, 1] = sum(cond)
        final_matrix = np.vstack((final_matrix, matrix))
        
    return final_matrix



def coincidences(ped1, ped2, dataframe, current_year, colname='Pedigreename'): 
    """
    Calculates the Number of Coincidences (N_ij) and normalized proportion 
    (expressed as percentage) between two pedigrees.

    This function implements the coincidence-based labeling heuristic defined 
    in Section 2 B.2 (Step 2) of Rosito et al. (2025).

    The process is:
    1. Compute the Family Feature Matrices M_i and M_j using compute_matrix.
    2. Compute the Coincidence Matrix C_ij = min(M_i, M_j) (element-wise minimum).
    3. Define the raw count N_ij as the sum of all elements of C_ij. 
    4. The normalized proportion is N_ij divided by the mean number of 
       individuals in the two pedigrees.

    Parameters
    ----------
    ped1 : str
        The unique identifier of the first pedigree.
    ped2 : str
        The unique identifier of the second pedigree.
    dataframe : pd.DataFrame
        The full dataset.
    current_year : int
        The fixed reference year for feature matrix calculation.
    colname : str, optional
        The name of the column containing the pedigree ID. 
        (Default is 'Pedigreename')

    Returns
    -------
    np.ndarray
        A 1D array of two elements: 
        [0]: Number of Coincidences (Nij, float).
        [1]: Normalized coincidence proportion (percentage, float)
    """

    mat1 = compute_matrix(ped1, dataframe, current_year, colname)
    mat2 = compute_matrix(ped2, dataframe, current_year, colname)
    coin = np.sum(np.minimum(mat1, mat2))
    nind1 =len(dataframe.loc[dataframe[colname] == ped1,])
    nind2 = len(dataframe.loc[dataframe[colname] == ped2,])
    nind = (nind1 + nind2)/2
    res = np.array((coin, coin/nind * 100))
    return res

def coincidences_pairs(pairs, dataframe, current_year, colname = 'Pedigreename'):
    """
    Calculates the Number of Coincidences (N_ij) and normalized proportion for 
    a list of candidate pedigree pairs.

    This function iterates over a list of pedigree pairs, calling 
    `coincidences` for each pair to calculate N_ij and the normalized proportion 
    (percentage). 
    These values are used in the labeling step (Step 2) in Section 2 B.2 of 
    Rosito et al. (2025).

    Parameters
    ----------
    pairs : array-like (list of lists or np.ndarray)
        A list where each element is a tuple (ped_A, ped_B).
    dataframe : pd.DataFrame
        The full dataset.
    current_year : int
        The fixed reference year for feature matrix calculation.
    colname : str, optional
        The name of the column containing the pedigree ID. 
        (Default is 'Pedigreename')


    Returns
    -------
    tuple of list
        A tuple containing two lists:
        [0]: `ncoincidences` (list of float) - Number of Coincidences (Nij)
        [1]: `proportions` (list of float) - Normalized coincidence proportions 
             for each pair.
    """

    ncoincidences = []
    proportions = []    
    for f in pairs:
        coin, prop = coincidences(f[0], f[1], dataframe, current_year, colname)
        ncoincidences.append(coin)
        proportions.append(prop)
        
    return ncoincidences, proportions
