# graph_construction/graph_builder.py

import networkx as nx
import pandas as pd

def build_pruned_graphs(df):
    """
    Builds bipartite graphs for each word in the DataFrame.
    """
    graphs = {}
    for word in df["word"].unique():
        G = nx.Graph()
        word_df = df[df["word"] == word]
        for _, row in word_df.iterrows():
            sentence, substitute = row["sentence"], row["replacement"]
            G.add_node(substitute, type=0)
            G.add_node(sentence, type=1)
            G.add_edge(substitute, sentence, weight=row["confidence"] / 10)
        graphs[word] = G
    return graphs