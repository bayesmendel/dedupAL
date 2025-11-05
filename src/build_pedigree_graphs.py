import numpy as np
import networkx as nx
import pandas as pd
from src import config
from collections import OrderedDict

def build_pedigree_graphs(dataframe, current_year, colname ='Pedigreename'):
    """
    Constructs a networkx graph for each unique pedigree in the DataFrame, 
    populating nodes with calculated and existing attributes required for 
    graph-based feature extraction (Rosito et al. 2025, Appendix A).

    The function prepares all necessary node attributes required by get_features(). 
    Edges (parental relationships) must be derived from existing columns 
    (e.g., FatherID, MotherID) if available.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The full dataset.
    current_year : int
        The fixed reference year for calculating 'age_at_present'.
    colname : str, optional
        The name of the column containing the pedigree ID 
        (Default is 'Pedigreename').

    Returns
    -------
    OrderedDict
        A dictionay where keys are pedigree IDs (str) and values are 
        networkx.DiGraph objects.
    """

    cancer_names = config.CANCER_NAMES
    
    # Prepare DataFrame and calculate 'age_at_present'
    df_copy = dataframe.copy()
    df_copy['age_at_present'] = current_year - df_copy['YearOfBirth']

    pedigree_graphs = OrderedDict()

    # Group by pedigree and build graphs
    for ped_id, ped_data in df_copy.groupby(colname):
        
        # Initialize a directed graph (standard for pedigree analysis)
        G = nx.DiGraph() 
        upns = np.array(df_copy.loc[df_copy['Pedigreename'] == ped_id, "UPN"])

        # Iterate through individuals (rows) to build nodes
        for _, row in ped_data.iterrows():

            node_attributes = {
                'age_at_present': row['age_at_present'],
                'Sex': row['Sex'], 
                'isDead': row['isDead'],
                'HadCancer': row['HadCancer'],
                'Presult': row['Presult']
            }

            for cancer in cancer_names:
                if cancer in ped_data.columns:
                    node_attributes[cancer] = row[cancer]
                else:
                    node_attributes[cancer] = 0

            G.add_node(row['UPN'], **node_attributes)

            if not pd.isna(row['Mother.ID']) and not row['Mother.ID'] == 0 and row['Mother.ID'] in upns:
                G.add_edge(row['Mother.ID'], row['UPN'])
            if not pd.isna(row['Father.ID']) and not row['Father.ID'] == 0 and row['Father.ID'] in upns:
                G.add_edge(row['Father.ID'], row['UPN'])

        pedigree_graphs[ped_id] = G

    return pedigree_graphs