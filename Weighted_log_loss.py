import numpy as np

def weighted_log_loss(y_true, y_pred_proba, classes):
    """
    Compute the weighted log loss for multi-class classification.

    Parameters:
    y_true (array-like): True class labels (should be an array or series).
    y_pred_proba (array-like): Predicted probabilities for each class.
    classes (array-like): List of class labels.

    Returns:
    Weighted log loss value
    """
    # Ensure y_true is a NumPy array
    y_true = np.array(y_true)

    # Convert y_true to one-hot encoding
    y_true_onehot = (y_true.reshape(-1, 1) == np.array(classes).reshape(1, -1))
    y_true_onehot = y_true_onehot.astype(np.float32)

    # Check if the number of classes in y_pred_proba and y_true_onehot match
    num_classes = len(classes)
    if y_pred_proba.shape[1] != num_classes:
        print(f"Warning: Number of classes mismatch! Expected {num_classes} classes, but got {y_pred_proba.shape[1]} classes.")
        
        # Ensure y_pred_proba matches the expected number of classes by trimming or padding
        if y_pred_proba.shape[1] > num_classes:
            y_pred_proba = y_pred_proba[:, :num_classes]  # Trim extra columns
        elif y_pred_proba.shape[1] < num_classes:
            padding = np.zeros((y_pred_proba.shape[0], num_classes - y_pred_proba.shape[1]))
            y_pred_proba = np.hstack([y_pred_proba, padding])  # Pad with zeros if too few classes

    # Calculate the class frequencies for weighted loss calculation
    class_frequencies = np.sum(y_true_onehot, axis=0) / len(y_true)
    
    # Compute log loss
    epsilon = 1e-15  # To avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1.0 - epsilon)
    
    log_loss = -np.sum(y_true_onehot * np.log(y_pred_proba)) / len(y_true)
    
    # Weighted log loss
    weighted_loss = np.sum(class_frequencies * log_loss)
    
    return weighted_loss