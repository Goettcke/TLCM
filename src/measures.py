# Since we don't want to keep reinstalling the DS-Pipe package, some measures are thrown in here during the development.
import numpy as np
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import math
from collections import Counter
from ds_pipe.utils.helper_functions import key_with_min_val, key_with_max_val
from sklearn.metrics import silhouette_score, silhouette_samples
from gower import gower_matrix
import networkx as nx 
from itertools import starmap
from scipy.spatial.distance import pdist
import statistics
from imblearn.under_sampling import TomekLinks


class IllegalArgumentError(ValueError): 
    def __init__(self, illegal_argument, options):
        self.illegal_argument = illegal_argument
        self.message = f"{illegal_argument} not within these options: {', '.join(options)}" 
        super().__init__(self.message)
    
def gower_metric(x,y): 
    g_matrix = gower_matrix(np.array([x,y]))
    return g_matrix[0,1]


def complexity_adjusted_imbalance_ratio(dataset,k=1.5, r=2, metric="Euclidean") -> float:
    """
    """
    assert len(set(dataset.target)) == 2, "Problem should be dichotomous"
    counts = Counter(dataset.target).values()
    ir = max(counts)/min(counts)
    if metric == "Euclidean": 
        return ir*math.pow((-(silhouette_score(X=dataset.data, labels=dataset.target)-k)),r)
    elif metric == "Gower": 
        dist = gower_matrix(dataset.data)
        return ir*math.pow((-(silhouette_score(X=dist, labels=dataset.target, metric="precomputed")-k)),r)
    else: 
        raise IllegalArgumentError(metric, ["Euclidean", "Gower"])


def new_cair(dataset,k=1.5, r=2, metric="Euclidean") -> float:
    """
    """
    assert len(set(dataset.target)) == 2, "Problem should be dichotomous"
    counts = Counter(dataset.target).values()
    ir = max(counts)/min(counts)
    sil_dist = silhouette_samples(dataset.data, dataset.target) 
    # Then we split the dataset into its different classes
    label_indices = [np.where(dataset.target == label)[0] for label in set(dataset.target)] 
    variations = [-(statistics.variance(sil_dist[label_indices[i]])-1) for i in range(len(label_indices))] 
    means = [np.mean(sil_dist[label_indices[i]]) for i in range(len(label_indices))] 
    #print(variations)
    return ir*np.mean(variations)


def tlcm(dataset): 
    tl = TomekLinks()
    n = len(dataset.target)
    counts = Counter(dataset.target).values()
    ir = max(counts)/min(counts)
    _,y_res = tl.fit_resample(dataset.data, dataset.target)
    return 1-(len(y_res)/n)


def tlcm_2(dataset): 
    """
    The idea is, that we take the proportion of the Tomek links, from the minority class in the dataset.
    """ 
    counts = Counter(dataset.target)
    c_min = key_with_min_val(counts)
    min_indices = [index for index, yi in enumerate(dataset.target) if yi == c_min]
    nbrs = NearestNeighbors(n_neighbors=2).fit(dataset.data)
    distances, nbr_indices = nbrs.kneighbors(dataset.data)
    c_min_tomek_links = 0
    for index in min_indices: 
        nn_index = nbr_indices[index][1] # index 0 is the point it self
        if dataset.target[nn_index] != c_min: 
            if nn_index == nbr_indices[nn_index][0]: 
                # Then they are mutual nearest neighbors of different classes i.e. Tomek Links
                c_min_tomek_links += 1 
    return c_min_tomek_links/counts[c_min] 

def tlcm_2_mean(dataset): 
    """
    The idea is, that we take the macro averaged proportion of the Tomek links, from all classes in the dataset.
    """ 
    counts = Counter(dataset.target)
    tls = []
    
    nbrs = NearestNeighbors(n_neighbors=2).fit(dataset.data)
    distances, nbr_indices = nbrs.kneighbors(dataset.data)
    
    for label in counts.keys(): 
        indices = [index for index, yi in enumerate(dataset.target) if yi == label]
        tomek_link_count = 0
        for index in indices: 
            nn_index = nbr_indices[index][1] # index 0 is the point it self
            if dataset.target[nn_index] != label: 
                if nn_index == nbr_indices[nn_index][0]: 
                    # Then they are mutual nearest neighbors of different classes i.e. Tomek Links
                    tomek_link_count += 1 
        tls.append(tomek_link_count/counts[label])
        return np.mean(tls) 


#TODO Gower distance is not implemented here yet.
def n_3(dataset, metric="Euclidean"): 
    loo = LeaveOneOut()
    loo.get_n_splits(dataset.data)
    accuracy_scores = []

    for train_index, test_index in loo.split(dataset.data): 
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(dataset.data[train_index], dataset.target[train_index])
        y_pred = clf.predict(dataset.data[test_index])
        accuracy_scores.append(accuracy_score(y_true = dataset.target[test_index],y_pred=y_pred)) 

    return 1-np.mean(accuracy_scores)


def n_3_imb(dataset, metric="Euclidean", label="min"): 
    #from time import time 
    #start_time = time() 
    from sklearn.neighbors import KDTree
    counts = Counter(dataset.target)
    if label == "min":  
        label = key_with_min_val(counts)

    if metric.lower() ==  "gower": 
        nn = NearestNeighbors(n_neighbors=2, metric=gower_metric,n_jobs=-1) 
        nn.fit(dataset.data)
        knns = nn.kneighbors(dataset.data,return_distance=False)
    elif metric.lower() == "euclidean": 
        tree = KDTree(dataset.data, leaf_size=2) 
        knns = tree.query(dataset.data, k=2, return_distance=False)
    
    no_different_labels = 0
    for index,knn_set in enumerate(knns): 
        nn_index = knn_set[1]
        if dataset.target[index] == label and dataset.target[nn_index] != label: 
            no_different_labels += 1
    
    return no_different_labels/counts[label]


def n_3_imb_mean(dataset, metric="Euclidean"):
    li = [(dataset,metric, label) for label in list(set(dataset.target))]
    return np.mean(list(starmap(n_3_imb, li)))


#This implementation  could perhaps be optimized with building a kNN graph instead of the complete graph, which takes forever.


def n_1_imb(dataset, metric="Euclidean",label="min"):
    """
    # Keyword arguments: 
       
    - dataset: A bunch dataset containing dataset.data and dataset.target
    - metric: The metric used for computing dissimilarity between instances 
    - label: The label for which to compute the measure. As this is a per label measure, either pick the class, or use 'min' for minority class, to compute for the smallest minority class. 
    """ 
    counter = Counter(dataset.target)
    if label == "min": 
        label = key_with_min_val(counter)
    
    G = nx.Graph()
    G.add_nodes_from(range(len(dataset.target)))
    nx.set_node_attributes(G,{i: dataset.target[i] for i in range(len(dataset.target))}, name="label")
    
    # Build the complete graph
    if metric.lower() == "euclidean": 
        for i in range(G.number_of_nodes()): 
            for j in range(i,G.number_of_nodes()): 
                G.add_edge(i,j,weight = np.linalg.norm(dataset.data[i]-dataset.data[j]))
    
    elif metric.lower() == "gower": 
        g_matrix = gower_matrix(dataset.data)   
        for i in range(G.number_of_nodes()): 
            for j in range(i,G.number_of_nodes()): 
                G.add_edge(i,j,weight = g_matrix[i,j])

    # Compute the minimum spanning tree
    T = nx.minimum_spanning_tree(G)

    # Finding the number of differently labeled neighbors in the MST 
    no_differently_labeled = 0
    for node in T.nodes: 
        if dataset.target[node] == label: 
            for edge in T.edges(node):
                if dataset.target[edge[0]] != dataset.target[edge[1]]: 
                    no_differently_labeled += 1 
                    break

    return no_differently_labeled/counter[label]


def n_1_imb_mean(dataset, metric="Euclidean"):
    li = [(dataset,metric, label) for label in list(set(dataset.target))]
    return np.mean(list(starmap(n_1_imb, li)))

def imbalance_ratio(dataset): 
    counts = Counter(dataset.target)
    c_min = key_with_min_val(counts)
    c_maj = key_with_max_val(counts)
    return counts[c_maj]/counts[c_min]


def degOver(dataset):
    """ 
    Binary implementation of 
    """
    counts = Counter(dataset.target)
    c_min = key_with_min_val(counts)
    c_maj = key_with_max_val(counts)
    nn = NearestNeighbors(n_neighbors=5) 
    nn.fit(dataset.data)
    _, indices = nn.kneighbors(dataset.data) # _, is distances
    n_min_over,n_maj_over = 0,0
    for i, neighbor_indices in enumerate(indices):
        if dataset.target[i] == c_min and c_maj in [dataset.target[j] for j in neighbor_indices]:
            n_min_over += 1
        elif dataset.target[i] == c_maj and c_min in [dataset.target[j] for j in neighbor_indices]:
            n_maj_over += 1
    
    return (n_min_over + n_maj_over)/len(dataset.target)

def degIR(dataset):
    """ 
    Binary implementation of 
    """
    counts = Counter(dataset.target)
    n = len(dataset.target)
    c_min = key_with_min_val(counts)
    c_maj = key_with_max_val(counts)
    return 1-(counts[c_min]/(n/2))




