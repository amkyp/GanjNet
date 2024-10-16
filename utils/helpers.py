# utils/helpers.py

import pandas as pd
import numpy as np
import os
from collections import Counter
import json

def tag_sentences_with_communities(df, partition, G):
    """
    Tags sentences with predicted communities.
    """
    community_tagged_sentences = []
    for _, row in df.iterrows():
        community_id = partition.get(row['replacement'], -1)  # Default to -1 if not found
        weight = G[row['replacement']][row['sentence']]['weight'] if G.has_edge(row['replacement'], row['sentence']) else 0

        community_tagged_sentences.append({
            'word': row['word'],
            'sentence': row['sentence'],
            'replacement': row['replacement'],
            'predicted_sense': f"sense_{community_id}" if community_id != -1 else 'removed',
            'weight': weight,
        })
    return pd.DataFrame(community_tagged_sentences)

def most_frequent_community(group, MFC='majority'):
    """
    Determines the most frequent community in a group.
    """
    if MFC == 'majority':
        community_counts = Counter(group['predicted_sense'])
        most_common_community = community_counts.most_common(1)[0][0]
        return pd.Series({'predicted_sense': most_common_community})
    else:
        raise NotImplementedError("Only 'majority' method is implemented.")

def save_community_tagged(df, word, output_dir):
    """
    Saves the community-tagged DataFrame.
    """
    output_file = os.path.join(output_dir, f"{word}_community_tagged.csv")
    df.to_csv(output_file, index=False, encoding='utf-8')

def load_ground_truth_validation(word, ground_truth_dir):
    """
    Loads the ground truth validation data.
    """
    filepath = os.path.join(ground_truth_dir, f"{word}_results.xlsx")
    if os.path.exists(filepath):
        return pd.read_excel(filepath)
    else:
        print(f"Ground truth for '{word}' not found at '{filepath}'.")
        return pd.DataFrame()

def align_ground_truth_with_predictions(ground_truth, predictions, on=['word', 'sentence']):
    """
    Aligns the ground truth and predicted DataFrames on specified columns.
    """
    aligned = pd.merge(ground_truth, predictions, on=on, suffixes=('_gt', '_pred'))
    return aligned

def save_communities_to_json(word_communities, filename):
    """
    Saves the word communities to a JSON file.
    """
    def convert_to_json(obj):
        if isinstance(obj, dict):
            return {convert_to_json(key): convert_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj

    json_output = json.dumps(word_communities, default=convert_to_json, indent=4)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(json_output)