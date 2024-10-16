# main.py

import os
from data_processing.data_loader import load_and_preprocess_data, prepare_gcn_input, normalize_features
from graph_construction.graph_builder import build_pruned_graphs
from models.gcn_model import train_model, save_model, CustomGraphDataset
from clustering.hierarchical_clustering import perform_hierarchical_cl
from community_detection.community_detector import perform_community_detection_leiden
from evaluation.metrics import evaluate_all_words, write_evaluation_results, calculate_overall_results
from utils.helpers import tag_sentences_with_communities, most_frequent_community, save_community_tagged, load_ground_truth_validation, align_ground_truth_with_predictions
import numpy as np
import pandas as pd

# Define constants
PROCESSED_DATA_DIR = "processed_data-revised"
OUTPUT_DIR = "output-features-Aug-25"
GROUND_TRUTH_DIR = "ground_truth"

def process_word(word):
    """
    Processes a single word: trains the model, performs clustering, and evaluates.
    """
    df = load_train_set(word, PROCESSED_DATA_DIR)
    if df.empty:
        print(f"Skipping word '{word}' due to missing data.")
        return None, None

    features, labels, G, train_mask, val_mask, test_mask, test_df = load_and_preprocess_data(word, PROCESSED_DATA_DIR)
    if features is None:
        print(f"Skipping word '{word}' due to missing data.")
        return None, None
    feature_matrix, label_vector, adj_matrix = prepare_gcn_input(features, labels, G)
    feature_matrix = normalize_features(feature_matrix)

    num_classes = 12  # Adjust based on your data

    train_dataset = CustomGraphDataset(adj_matrix, feature_matrix, label_vector, train_mask)
    val_dataset = CustomGraphDataset(adj_matrix, feature_matrix, label_vector, val_mask)
    test_dataset = CustomGraphDataset(adj_matrix, feature_matrix, label_vector, test_mask)
    print(f'Feature matrix shape: {feature_matrix.shape}, Adjacency matrix shape: {adj_matrix.shape}, Label vector shape: {label_vector.shape}')

    # Option to choose between community detection methods
    use_leiden = True  # Set to False to use hierarchical clustering

    if use_leiden:
        # Perform community detection using Leiden algorithm
        community_partition = perform_community_detection_leiden(G)
    else:
        # Train GCN model and perform hierarchical clustering
        model, history, embeddings = train_model(train_dataset, val_dataset, num_classes, word)
        save_model(model, word, OUTPUT_DIR)

        node_list = list(G.nodes())
        substitute_node_indices = [i for i, node in enumerate(node_list) if G.nodes[node]['type'] == 0]
        clustering_labels = perform_hierarchical_cl(embeddings, substitute_node_indices, word, OUTPUT_DIR)

        substitute_nodes = [node_list[i] for i in substitute_node_indices]
        community_partition = {node: label for node, label in zip(substitute_nodes, clustering_labels)}

    # Tag sentences with communities and evaluate
    community_tagged_df = tag_sentences_with_communities(df[df['word'] == word], community_partition, G)
    community_tagged_df = community_tagged_df.groupby(['word', 'sentence']).apply(most_frequent_community, MFC='majority').reset_index()

    save_community_tagged(community_tagged_df, word, OUTPUT_DIR)

    # Load ground truth and evaluate
    ground_truth = load_ground_truth_validation(word, GROUND_TRUTH_DIR)
    if ground_truth.empty:
        print(f"No ground truth data for word '{word}'.")
        return community_partition, G

    # Align predictions with ground truth
    aligned_data = align_ground_truth_with_predictions(ground_truth, community_tagged_df)

    # Evaluate the refined output
    labels_true = aligned_data['label']
    labels_pred = aligned_data['predicted_sense']
    v_measure_score, h, c = v_measure(labels_true, labels_pred)
    print(f"Metrics for {word}: V-Measure={v_measure_score:.4f}, Homogeneity={h:.4f}, Completeness={c:.4f}")

    return community_partition, G

def main():
    """
    Main function to process all words.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    words_to_process = [f for f in os.listdir(PROCESSED_DATA_DIR) if os.path.isdir(os.path.join(PROCESSED_DATA_DIR, f))]
    all_word_communities = {}
    for word in words_to_process:
        print(f"\nProcessing word: {word}")
        word_communities, word_graph = process_word(word)
        if word_communities:
            all_word_communities[word] = word_communities

    evaluation_results = evaluate_all_words(words_to_process, OUTPUT_DIR)
    overall_results = calculate_overall_results(evaluation_results)
    evaluation_results = pd.concat([evaluation_results, overall_results], ignore_index=True)
    write_evaluation_results(evaluation_results, OUTPUT_DIR, filename='evaluation_results.csv')
    print("Evaluation completed. Results saved in the output directory.")

if __name__ == "__main__":
    main()
