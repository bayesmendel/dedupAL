import numpy as np
import networkx as nx
from src import config

def get_features(G):
    """
    Extracts a feature vector from a pedigree graph representation.

    This function computes both structural and node-level features for a single 
    pedigree graph, as described in Appendix A of Rosito et al. (2025).

    The output includes:
      - Structural metrics (number of nodes, edges, density, etc.)
      - Node-level averages (average age, proportion of dead, proportion of 
      cancer affected individuals, etc.)

    Parameters
    ----------
    G : networkx.DiGraph
        Pedigree graph. Nodes must contain the following attributes:
        'age_at_present' and binary columns for 'Sex', 'isDead', 
        'HadCancer', 'Presult' (positive test result), and one binary 
        column per cancer  type (e.g., 'Adrenal', 'Brain', 'Breast', ...).

    Returns
    -------
    np.ndarray
        A 1D NumPy array with the following features:
        [num_nodes, num_edges, density, avg_degree, avg_shortest_path_length,
         mean_Sex, mean_HasCancer, mean_CurAge, mean_isDead, mean_Presult,
         mean_<Cancer1>, ..., mean_<CancerK>, num_components]
    """

    # --- Structural Features ---
    # Number of nodes
    num_nodes = G.number_of_nodes()
    # Number of edges
    num_edges = G.number_of_edges()
    # Graph density   
    density = nx.density(G)
    # Average degree
    avg_degree = sum(dict(G.degree()).values()) / num_nodes

    G_undirected = G.to_undirected()
    # Average shortest path length
    if nx.is_connected(G_undirected):
        avg_shortest_path_length = nx.average_shortest_path_length(G_undirected)
    else:
        avg_shortest_path_length = -1
    # Number of conneceted components
    num_components = nx.number_connected_components(G_undirected)        

    def safe_mean(values):
        arr = np.array(values, dtype=float)
        return np.nan if len(arr) == 0 or np.all(np.isnan(arr)) else np.nanmean(arr)

    # --- Node-Based Features ---
    # Sex == 0/1:  0 = female, 1 = male 
    node_Sex = np.array([G.nodes[node]['Sex'] for node in G.nodes()])
    # Any cancer
    node_HadCancer = np.array([G.nodes[node]['HadCancer'] for node in G.nodes()])  
    # age_at_present = current year - year of birth    
    node_age_at_present = np.array([G.nodes[node]['age_at_present'] for node in G.nodes()])
    # Deceased status
    node_isDead = np.array([G.nodes[node]['isDead'] for node in G.nodes()])
    # Positive test result    
    node_Presult = np.array([G.nodes[node]['Presult'] for node in G.nodes()])  

    avg_Sex = safe_mean(node_Sex)
    avg_HadCancer = safe_mean(node_HadCancer)
    avg_age_at_present = safe_mean(node_age_at_present)
    avg_isDead = safe_mean(node_isDead)
    avg_Presult = safe_mean(node_Presult)

    # Cancer-specific averages
    avg_cancers = []
    for cancer in config.CANCER_NAMES:
        cancer_values = np.array([G.nodes[node][cancer] for node in G.nodes()])
        avg_cancers.append(safe_mean(cancer_values))

    features = np.array(
        [num_nodes, num_edges, density, avg_degree, avg_shortest_path_length,
         avg_Sex, avg_HadCancer, avg_age_at_present, avg_isDead, avg_Presult,
         *avg_cancers, num_components]
    )

    return features

def extract_features(G1, G2):
    """
    Calculates the element-wise absolute difference between the feature vectors 
    of two pedigree graphs.

    This generates the feature vector used for the machine learning classifier 
    described in Section 2 B.3 (Step 3) by quantifying the structural and 
    node-level differences between the two graphs.

    The difference is computed as |F1 - F2| where Fi = get_features(Gi)
    If any element in F1 or F2 is NaN, the corresponding difference is 
    set to NaN.

    Parameters
    ----------
    G1 : networkx.Graph
        The first pedigree graph.
    G2 : networkx.Graph
        The second pedigree graph.

    Returns
    -------
    np.ndarray
        A 1D NumPy array representing the absolute difference between the 
        feature vectors of G1 and G2. The length is equal to the number of 
        features extracted by get_features(G).
    """
    
    # Get features for both graphs
    features1 = get_features(G1) 
    features2 = get_features(G2) 

    features1 = np.asarray(features1)
    features2 = np.asarray(features2)

    mask = np.isnan(features1) | np.isnan(features2)
    diff = np.abs(features1 - features2)
    result = np.where(mask, np.nan, diff)
    
    return result