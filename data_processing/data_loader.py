import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from graph_construction.graph_builder import build_pruned_graphs
import networkx as nx

def load_train_set(word, processed_data_dir):
    """
    Loads the training set for a given word.
    """
    filepath = os.path.join(processed_data_dir, word, f"{word}_train_set-revised.xlsx")
    if os.path.exists(filepath):
        df = pd.read_excel(filepath, engine='openpyxl')
        return df
    else:
        print(f"Training data for '{word}' not found at '{filepath}'.")
        return pd.DataFrame()

def load_and_preprocess_data(word, processed_data_dir):
    """
    Loads and preprocesses data for a given word.
    """
    df = load_train_set(word, processed_data_dir)
    if df.empty:
        return None, None, None, None, None, None, None
    train_df = df[df['tag'] == 'train']
    test_df = df[df['tag'] == 'test']
    df["label_vector"] = df["label"].apply(create_label_vector)

    G = build_pruned_graphs(df)[word]

    train_mask, val_mask, test_mask = create_masks(G, train_df, test_df)
    features = extract_node_features(G)
    labels = {node: df[df["sentence"] == node]["label_vector"].iloc[0] for node in G.nodes() if G.nodes[node]["type"] == 1}

    return features, labels, G, train_mask, val_mask, test_mask, test_df

def extract_node_features(G):
    """
    Extracts features for each node in the graph.
    """
    features = {}
    degree_centrality = nx.degree_centrality(G)
    for node in G.nodes():
        degree = G.degree(node)
        node_type = G.nodes[node]["type"]
        features[node] = [node_type, degree, degree_centrality[node]]
    return features

def create_masks(G, train_df, test_df):
    """
    Creates masks for training, validation, and testing sets.
    """
    train_mask = np.zeros(len(G.nodes()), dtype=bool)
    val_mask = np.zeros(len(G.nodes()), dtype=bool)
    test_mask = np.zeros(len(G.nodes()), dtype=bool)
    node_list = list(G.nodes())
    node_indices = {node: idx for idx, node in enumerate(node_list)}
    for node in train_df['sentence'].values:
        if node in node_indices:
            train_mask[node_indices[node]] = True
    for node in test_df['sentence'].values:
        if node in node_indices:
            test_mask[node_indices[node]] = True
    val_indices = np.random.choice(np.where(train_mask)[0], size=int(0.2 * np.sum(train_mask)), replace=False)
    val_mask[val_indices] = True
    train_mask[val_indices] = False
    return train_mask, val_mask, test_mask

def create_label_vector(label, num_senses=12):
    """
    Converts a label into a one-hot encoded vector.
    """
    vector = [0] * num_senses
    if pd.notna(label) and label.startswith("sense_"):
        sense_num = int(label.split("_")[1])
        if sense_num >= num_senses:
            print(f"Warning: Sense number {sense_num} is greater than the number of senses {num_senses}")
            return vector
        vector[sense_num] = 1
    return vector

def prepare_gcn_input(features, labels, G, num_senses=12):
    """
    Prepares input data for the GCN model.
    """
    node_list = list(G.nodes())
    feature_matrix = np.array([features[node] for node in node_list])
    label_vector = np.array([labels.get(node, [0] * num_senses) for node in node_list])
    adj_matrix = nx.adjacency_matrix(G).toarray()
    return feature_matrix, label_vector, adj_matrix

def normalize_features(feature_matrix):
    """
    Normalizes the feature matrix.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(feature_matrix)
