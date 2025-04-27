import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from load_data import load_data
from show_result import show_result

# RandomForest
def run_experiment(use_smote, use_class_weight, description):
    print(f"\n{'='*50}")
    print(f"Running Experiment: {description}")
    print('='*50)
    
    # loading data
    X_train, y_train, X_test1, X_test2, y_test2 = load_data()
    
    # Divide the training set and the validation set hierarchically
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42,
        stratify=y_train
    )
    
    # SMOTE
    if use_smote:
        class_counts = np.bincount(y_train_part)
        min_samples = np.min(class_counts[class_counts > 0])
        k_neighbors = min(3, min_samples-1) if min_samples > 1 else 1
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_part, y_train_part = smote.fit_resample(X_train_part, y_train_part)
    
    # Category weight calculation
    class_weights = None
    if use_class_weight:
        classes = np.unique(y_train_part)
        weights = compute_class_weight('balanced', classes=classes, y=y_train_part)
        class_weights = dict(zip(classes, weights))
    
    # Construct the random forest model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        class_weight=class_weights,
        random_state=42,
        criterion='gini',
        n_jobs=-1
    )
    
    # model training
    model.fit(X_train_part, y_train_part)
    
    show_result(model, X_val, y_val, X_test2, y_test2, True)
    
def main():
   # 1. RandomForest results without imbalanced processing of the data
    run_experiment(use_smote=False, use_class_weight=False, description="Baseline RandomForest (no SMOTE & class_weight)")

    # 2. The data only undergoes the RandomForest results under SMOTE preprocessing (without using class_weight)
    run_experiment(use_smote=True, use_class_weight=False, description="RandomForest + SMOTE (no class_weight)")

    # 3. The data only performs the RandomForest results under class_weight='balanced' (without using SMOTE)
    run_experiment(use_smote=False, use_class_weight=True, description="RandomForest + class_weight='balanced' (no SMOTE)")

    # 4. The data undergoes SMOTE processing and the RandomForest result under class_weight='balanced' is added
    run_experiment(use_smote=True, use_class_weight=True, description="RandomForest + SMOTE + class_weight='balanced'")

if __name__ == "__main__":
    main()