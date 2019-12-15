import networkx as nx
import numpy as np
from collections import Counter
from operator import sub
from functools import reduce


def __get_strongly_components(G):
    return nx.strongly_connected_components(G)


def __compute_diameter(graph):
    return nx.diameter(graph)


def __largest_strongly_connected_set(G):
    largest_strongly_connected_set = max(nx.strongly_connected_components(G), key=len)
    return nx.subgraph(G, largest_strongly_connected_set)


def __mean_square_distances(true_values, predicted_values):
    if isinstance(true_values, dict) and isinstance(predicted_values, dict):
        master = {'predict': predicted_values, 'truth': true_values}
        distances = np.array(reduce(sub, map(Counter, master.values())))
    elif isinstance(true_values, np.ndarray) and isinstance(predicted_values, np.ndarray):
        distances = true_values - predicted_values
    else:
        raise TypeError("The predicted and true values should be passed both as Python dictionaries or both as Numpy "
                        "arrays.")
    squared_dists = np.matmul(distances, distances.T)
    msd = np.sqrt(squared_dists / len(distances))
    return msd
