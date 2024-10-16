# evaluation/metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_score, completeness_score, normalized_mutual_info_score

def v_measure(labels_true, labels_pred):
    """
    Computes the V-measure between true labels and predicted labels.
    """
    h = homogeneity_score(labels_true, labels_pred)
    c = completeness_score(labels_true, labels_pred)
    v = 2 * (h * c) / (h + c) if (h + c) > 0 else 0
    return v, h, c

def paired_f_score(true_labels, predicted_labels, sentences):
    """
    Computes the paired F-score between true and predicted labels.
    """
    pairs_true = set()
    for i in range(len(true_labels)):
        for j in range(i + 1, len(true_labels)):
            if true_labels[i] == true_labels[j]:
                pairs_true.add(tuple(sorted((i, j))))

    pairs_predicted = set()
    for i in range(len(predicted_labels)):
        for j in range(i + 1, len(predicted_labels)):
            if predicted_labels[i] == predicted_labels[j]:
                pairs_predicted.add(tuple(sorted((i, j))))

    intersection = pairs_true & pairs_predicted
    a = len(intersection)  # True positives
    b = len(pairs_true - intersection)  # False negatives
    c = len(pairs_predicted - intersection)  # False positives

    precision = a / (a + c) if a + c > 0 else 0
    recall = a / (a + b) if a + b > 0 else 0
    f_score = 2 * a / (2 * a + b + c) if 2 * a + b + c > 0 else 0

    return precision, recall, f_score

def bcubed_score(labels_true, labels_pred):
    """
    Computes the B-Cubed precision, recall, and F-score.
    """
    true_clusters = {}
    pred_clusters = {}

    for i, (true_label, pred_label) in enumerate(zip(labels_true, labels_pred)):
        true_clusters.setdefault(true_label, set()).add(i)
        pred_clusters.setdefault(pred_label, set()).add(i)

    precision = []
    recall = []

    for i in range(len(labels_true)):
        true_cluster = true_clusters[labels_true[i]]
        pred_cluster = pred_clusters[labels_pred[i]]
        intersection_size = len(true_cluster & pred_cluster)
        precision.append(intersection_size / len(pred_cluster))
        recall.append(intersection_size / len(true_cluster))

    precision = np.mean(precision)
    recall = np.mean(recall)
    f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f_score

def evaluate_all_words(words_to_process, output_dir):
    """
    Evaluates the model across all words and saves the results.
    """
    results = []
    for word in words_to_process:
        ground_truth_file = os.path.join(output_dir, f'{word}-final-ground_truth.csv')
        model_output_file = os.path.join(output_dir, f'{word}-final-model_output.xlsx')
        if not os.path.exists(ground_truth_file) or not os.path.exists(model_output_file):
            print(f"Missing files for word '{word}', skipping evaluation.")
            continue
        ground_truth = pd.read_csv(ground_truth_file, encoding='utf-8')
        model_output = pd.read_excel(model_output_file)

        merged_data = pd.merge(ground_truth, model_output, on=['word', 'sentence'])
        true_labels = merged_data['label']
        predicted_labels = merged_data['predicted_sense']
        sentences = merged_data['sentence']

        precision, recall, paired_fscore = paired_f_score(true_labels, predicted_labels, sentences)
        v_measure_score, h, c = v_measure(true_labels, predicted_labels)
        bcubed_precision, bcubed_recall, bcubed_fscore = bcubed_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels, average_method='arithmetic')

        results.append({
            'word': word,
            'V-Measure': round(v_measure_score, 3),
            'Paired F-Score': round(paired_fscore, 3),
            'Homogeneity': round(h, 3),
            'Completeness': round(c, 3),
            'Precision': round(precision, 3),
            'Recall': round(recall, 3),
            'B-Cubed F-Score': round(bcubed_fscore, 3),
            'B-Cubed Precision': round(bcubed_precision, 3),
            'B-Cubed Recall': round(bcubed_recall, 3),
            'NMI': round(nmi, 3),
        })
    return pd.DataFrame(results)

def write_evaluation_results(results_df, output_dir, filename):
    """
    Writes evaluation results to a CSV file.
    """
    output_path = os.path.join(output_dir, filename)
    results_df.to_csv(output_path, index=False)
    print(f"Evaluation results saved to {output_path}")

def calculate_overall_results(results_df):
    """
    Calculates overall results across all words.
    """
    numeric_df = results_df.select_dtypes(include=[np.number])