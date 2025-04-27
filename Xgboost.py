import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from load_data import load_data
from show_result import show_result

## xgboost
def run_experiment(use_smote, use_class_weight, description):
    print(f"\n{'='*50}")
    print(f"Running Experiment: {description}")
    print('='*50)
    
    # load data
    X_train, y_train, X_test1,  X_test2, y_test2 = load_data()
    
    # Divide the training set and the validation set hierarchically
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42,
        stratify=y_train
    )
    
    # handle SMOTE
    if use_smote:
        class_counts = np.bincount(y_train_part)
        min_samples = np.min(class_counts[class_counts > 0])
        k_neighbors = min(3, min_samples-1) if min_samples > 1 else 1
        
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_part, y_train_part = smote.fit_resample(X_train_part, y_train_part)
    
    # calculate class_weight
    class_weights = None
    if use_class_weight:
        classes = np.unique(y_train_part)
        weights = compute_class_weight('balanced', classes=classes, y=y_train_part)
        weight_dict = dict(zip(classes, weights))
        class_weights = np.array([weight_dict[y] for y in y_train_part])  # Convert to list format
    
    # XGBoost modeling
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=28,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        scale_pos_weight='balanced',  # Automatic balancing category
        colsample_bytree=0.8,
        tree_method='gpu_hist',
        gpu_id=0,
        predictor='gpu_predictor',
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    # training model
    model.fit(
        X_train_part, y_train_part,
        sample_weight=class_weights,  
        eval_set=[(X_val, y_val)],
        verbose=0
    )
    
    # output specification
    show_result(model, X_val, y_val, X_test2, y_test2, True)
   
    # Save the prediction results
    if description == "xgboost + SMOTE + class_weight='balanced'":
        np.save(f'preds_1.npy', model.predict(X_test1))
        np.save(f'preds_2.npy', model.predict(X_test2[202:]))
def main():
    # 1. xgboost results without unbalanced processing of the data
    run_experiment(use_smote=False, use_class_weight=False, description="Baseline xgboost (no SMOTE & class_weight)")

    # 2. The data only undergoes the xgboost result under SMOTE preprocessing (without using class_weight)
    run_experiment(use_smote=True, use_class_weight=False, description="xgboost + SMOTE (no class_weight)")

    # 3. The data only performs the xgboost results under class_weight='balanced' (without using SMOTE)
    run_experiment(use_smote=False, use_class_weight=True, description="xgboost + class_weight='balanced' (no SMOTE)")

    # 4. The data undergoes SMOTE processing and the xgboost result under class_weight='balanced' is added
    run_experiment(use_smote=True, use_class_weight=True, description="xgboost + SMOTE + class_weight='balanced'")

if __name__ == "__main__":
    main()
