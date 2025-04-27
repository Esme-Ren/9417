import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from load_data import load_data
from show_result import show_result

def build_logistic_regression(use_class_weight=False):
    if use_class_weight:
        weight = 'balanced'
    else:
        weight = None
    model = LogisticRegression(
        class_weight=weight,
        multi_class='multinomial', 
        solver='lbfgs', 
        max_iter=5000, 
        random_state=42
    )
    return model

# ========================================
#Run the experiment. Parameter description:
# use_smote: Whether to perform SMOTE oversampling on the training data
# use_class_weight: Whether to set class_weight='balanced' in the model
# description: Experiment Description (for print output)
# ========================================
def run_experiment(use_smote, use_class_weight, description=""):
    print("\n" + "="*50)
    print(f"实验：{description}")
    print("="*50)
    
    # data loading
    X_train, y_train, X_test1, X_test2, y_test2 = load_data()
    
    # If SMOTE is chosen to be used, the training data will be oversampled
    if use_smote:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Divide the training set and the validation set (80% training, 20% validation)
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
   # Build a logistic regression model (depending on whether class_weight is used for construction)
    model = build_logistic_regression(use_class_weight)

    # Train the model on the training subset
    model.fit(X_train_part, y_train_part)

    # Output Metrics
    show_result(model, X_val, y_val, X_test2, y_test2, True)
# ========================================
# Main process: Run experiments under different schemes respectively
# ========================================
def main():
    # 1. LR results without unbalanced processing of the data
    run_experiment(use_smote=False, use_class_weight=False, description="Baseline LR (no SMOTE & class_weight)")

    # 2. The LR results of the data only under SMOTE preprocessing (without using class_weight)
    run_experiment(use_smote=True, use_class_weight=False, description="LR + SMOTE (no class_weight)")

    # 3. The data only performs the LR results under class_weight='balanced' (without using SMOTE)
    run_experiment(use_smote=False, use_class_weight=True, description="LR + class_weight='balanced' (no SMOTE)")

    # 4. The data undergoes SMOTE processing and the LR result under class_weight='balanced' is added
    run_experiment(use_smote=True, use_class_weight=True, description="LR + SMOTE + class_weight='balanced'")
if __name__ == "__main__":
    main()