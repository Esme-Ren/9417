import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from Weighted_log_loss import weighted_log_loss

def show_result(model, X_val, y_val, X_test2, y_test2, flag):
    if flag == True:  
        y_val_proba = model.predict_proba(X_val)
    else:
        y_val_proba = model.predict(X_val)
    y_val_pred = np.argmax(y_val_proba, axis=1)
    
    # Calculate the weighted logarithmic loss
    classes = np.unique(np.concatenate([y_val, y_val_pred]))
    val_loss = weighted_log_loss(y_val, y_val_proba, classes)
    
    print("\n=== Validate set performance ===")
    print(f"Weighted logarithmic loss: {val_loss:.4f}")
    print(f"Accurency: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Weighted F1: {f1_score(y_val, y_val_pred, average='weighted'):.4f}")
    print(f"Macro-F1: {f1_score(y_val, y_val_pred, average='macro'):.4f}")
    print("Classification report:")
    print(classification_report(y_val, y_val_pred, target_names=[f"Class {i}" for i in range(28)]))
    
    # Test set evaluation
    X_test2_labeled = X_test2[:202]
    if flag == True:
        y_test2_proba = model.predict_proba(X_test2_labeled)
    else:
        y_test2_proba = model.predict(X_test2_labeled)
    y_test2_pred = np.argmax(y_test2_proba, axis=1)
    
    # Calculate the weighted logarithmic loss (using only 202 labeled samples)
    test2_classes = np.unique(np.concatenate([y_test2, y_test2_pred]))
    test2_loss = weighted_log_loss(y_test2, y_test2_proba, test2_classes)
    
    
    print("\n=== Test2 set performance ===")
    print(f"Weighted logarithmic loss: {test2_loss:.4f}")
    print(f"Accurency: {accuracy_score(y_test2, y_test2_pred):.4f}")
    print(f"Weighted F1: {f1_score(y_test2, y_test2_pred, average='weighted', labels=test2_classes):.4f}")
    print(f"Macro-F1: {f1_score(y_test2, y_test2_pred, average='macro', labels=test2_classes):.4f}")
    print("Classification report:")
    test2_classes = [str(cls) for cls in test2_classes]

    print(classification_report(
        y_test2, y_test2_pred, 
        labels=test2_classes,
        target_names=test2_classes,
        zero_division=0  
    ))