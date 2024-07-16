import pandas as pd
import numpy as np

# Function to process each CSV file and print the required information
def process_csv(csv_file):
    print(f"Processing file: {csv_file}")
    df = pd.read_csv(csv_file)

    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Filter out rows with real labels 10 and 11
    df = df[(df['RealLabel y'] != 10) & (df['RealLabel y'] != 11)]

    # Number of classes
    num_classes = 9

    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))

    # Process each row to populate the confusion matrix
    for index, row in df.iterrows():
        true_label = int(row['RealLabel y'])
        predicted_label = int(row['PredictLabel y~'])
        if true_label != 10 and predicted_label != 10:
            confusion_matrix[true_label, predicted_label] += 1

    # Calculate precision and recall for each class
    precision_dict = {}
    recall_dict = {}
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = sum(confusion_matrix[:, i]) - tp
        fn = sum(confusion_matrix[i, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_dict[f'Class {i}'] = precision
        recall_dict[f'Class {i}'] = recall

        # Print number of instances and true positives for each class
        total_instances = sum(confusion_matrix[i, :])
        true_positives = tp
        print(f"Class {i}: Total Instances={int(total_instances)}, True Positives={int(true_positives)}, Precision={precision:.9f}, Recall={recall:.9f}")

    # Calculate F-beta scores for each class and beta value
    beta_values = [0.5, 1, 1.5]
    f_beta_values = {beta: [] for beta in beta_values}
    for beta in beta_values:
        for i in range(num_classes):
            f_beta = (1 + beta ** 2) * (precision_dict[f'Class {i}'] * recall_dict[f'Class {i}']) / ((beta ** 2 * precision_dict[f'Class {i}']) + recall_dict[f'Class {i}']) if ((beta ** 2 * precision_dict[f'Class {i}']) + recall_dict[f'Class {i}']) > 0 else 0
            f_beta_values[beta].append(f_beta)

    # Calculate average F-beta scores for each beta value
    avg_f_beta_values = {beta: np.mean(scores) for beta, scores in f_beta_values.items()}

    # Calculate CBA
    CBA_values = []
    for i in range(num_classes):
        a = confusion_matrix[i, i]
        b = sum(confusion_matrix[i, :])
        c = sum(confusion_matrix[:, i])
        max_bc = max(b, c)
        CBA = a / max_bc if max_bc > 0 else 0
        CBA_values.append(CBA)

    # Average CBA
    average_CBA = np.mean(CBA_values)

    # Calculate mGM
    a = np.prod([recall_dict[f'Class {i}'] for i in range(num_classes)])
    mGM = np.power(a, 1 / num_classes)

    # Calculate AVCC
    AVCC = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

    # Calculate CEN
    N = np.sum(confusion_matrix)
    CEN = 0
    for j in range(num_classes):
        pj = (np.sum(confusion_matrix[j, :]) + np.sum(confusion_matrix[:, j])) / (2 * N)
        CEN_j = 0
        for k in range(num_classes):
            if k != j:
                P_j_jk = confusion_matrix[j, k] / (np.sum(confusion_matrix[j, :]) + np.sum(confusion_matrix[:, j]))
                P_j_kj = confusion_matrix[k, j] / (np.sum(confusion_matrix[j, :]) + np.sum(confusion_matrix[:, j]))
                if P_j_jk > 0:
                    CEN_j -= P_j_jk * np.log2(P_j_jk / (num_classes - 1))
                if P_j_kj > 0:
                    CEN_j -= P_j_kj * np.log2(P_j_kj / (num_classes - 1))
        CEN += pj * CEN_j

    # Display the result
    print("AvF0.5\t\tAvF1\t\tAvF1.5\t\tCBA\t\tmGM\t\tAVCC\t\tCEN")
    print(f"{avg_f_beta_values[0.5]:.9f}\t{avg_f_beta_values[1]:.9f}\t{avg_f_beta_values[1.5]:.9f}\t{average_CBA:.9f}\t{mGM:.9f}\t{AVCC:.9f}\t{CEN:.9f}")
    print("\n")

# Process each CSV file
csv_files = ['Entry10-s1.csv', 'output_s1_copy.csv', 'output_s2_copy.csv', 'output_s3_copy.csv', 'output_s4_copy.csv' , 'output_s5_copy.csv']

for csv_file in csv_files:
    process_csv(csv_file)
