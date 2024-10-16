# clustering/hierarchical_clustering.py

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os

def perform_hierarchical_cl(embeddings, node_indices, word, output_dir):
    """
    Performs hierarchical clustering on embeddings and plots a dendrogram.
    """
    substitute_embeddings = embeddings[node_indices]

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='ward')
    clustering.fit(substitute_embeddings)

    linked = linkage(substitute_embeddings, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked, labels=[str(i) for i in node_indices])
    plt.title(f'Dendrogram for {word}')
    plt.xlabel('Node Index')
    plt.ylabel('Distance')
    plt.savefig(os.path.join(output_dir, f'{word}_dendrogram.png'))
    plt.close()

    return clustering.labels_
