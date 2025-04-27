import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from load_data import load_data

def diagnose_shift(X_train, y_train, X_test, y_test, alpha=0.05):
    """
    Compare and visualize label distributions (Label Shift) and
    test for feature distribution differences (Covariate Shift).
    """
    # 1. Label shift comparison
    print("=== Label Shift ===")
    train_label_counts = Counter(y_train)
    test_label_counts  = Counter(y_test)
    print("Training set label distribution:", train_label_counts)
    print("Test set label distribution:    ", test_label_counts)
    
    # Prepare labels and counts
    labels = sorted(set(train_label_counts) | set(test_label_counts))
    train_counts = [train_label_counts.get(lbl, 0) for lbl in labels]
    test_counts  = [test_label_counts.get(lbl, 0)  for lbl in labels]
    
    # Plot training label distribution
    plt.figure(figsize=(10, 4))
    plt.bar(labels, train_counts)
    plt.xlabel('Class Label')
    plt.ylabel('Count')
    plt.title('Training Set Label Distribution')
    plt.xticks(labels, rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Plot test label distribution
    plt.figure(figsize=(10, 4))
    plt.bar(labels, test_counts, color='orange')
    plt.xlabel('Class Label')
    plt.ylabel('Count')
    plt.title('Test Set Label Distribution')
    plt.xticks(labels, rotation=90)
    plt.tight_layout()
    plt.show()
    
    # 2. Covariate shift comparison: KS-test on each feature
    print("\n=== Covariate Shift (KS-test) ===")
    n_features = X_train.shape[1]
    p_values = []
    for i in range(n_features):
        _, p = ks_2samp(X_train[:, i], X_test[:, i])
        p_values.append(p)
    p_values = np.array(p_values)
    ratio = np.mean(p_values < alpha)
    print(f"{n_features} features in total, {ratio*100:.2f}% have significantly different distributions (p < {alpha})")

# Example usage
if __name__ == "__main__":
    # Assume load_data() returns X_train_scaled, y_train, X_test1_scaled, X_test2_scaled, y_test2
    X_train_scaled, y_train, _, X_test2_scaled, y_test2 = load_data()
    diagnose_shift(X_train_scaled, y_train, X_test2_scaled[:202], y_test2)
