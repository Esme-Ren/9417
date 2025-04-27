# COMP9417 Group Project – Machine Learning for Customer Feedback Classification

This project implements **three** models for classification, each of which used four different combinations of two imbalanced treatments to present the results:

1. **Logic Regression**  
    baseline LR / LR + SOMTE / LR + class_weight / LR + SMOTE & class_weight

2. **XGBoost**  
    baseline XGBoost / XGBoost + SOMTE / XGBoost + class_weight / XGBoost + SMOTE & class_weight
3. **Random Forest**  
    baseline RF / RF + SOMTE / RF + class_weight / RF + SMOTE & class_weight
---

## 📁 Project Structure
```text
project/
├── Group Project - Data-20250401/    # Store data files
│   ├── X_train.csv
│   ├── y_train.csv
│   ├── X_test_1.csv
│   ├── X_test_2.csv
│   └── y_test_2_reduced.csv
│
├── Weighted_log_loss.py   # Weighted logarithmic loss calculation
├── load_data.py           # Data loading and preprocessing
├── diagnose_shift.py      # Distribution offset diagnosis
├── show_result.py         # Result display and evaluation
├── Xgboost.py             # XGBoost Model Training and Experiment
├── random_forest.py       # Random Forest Model Training and Experiment
├── logistic_regression.py # Logic Regressoin Model Training and Experiment
├── preds_1.npy            # predictions for test set 1 (1000 unlabelled points)
├── preds_2.npy            # predictions for test set (the 1818 unlabelled points)
└── README.md              # This instruction document
```

---

## Install dependencies
```bash
pip install numpy pandas scikit-learn imbalanced-learn xgboost matplotlib scipy
```

---

## Configure data path
Place the data file（X_train.csv, y_train.csv, X_test_1.csv, X_test_2.csv, y_test_2_reduced.csv）in the 'Group Project - Data-20250401/' directory.
Modify the base_math parameter in load_data.py to the actual data path:
```bash
base_path = "your/data/path"  # 默认：'D:/p_code/unsw/comp9417/proj/Group Project - Data-20250401'
```

---

##  How to Run

### XGBoost
```bash
python src/xgboost.py
```

---


### Rabdon Forest
```bash
python src/random_forest.py
```

---

### Logic Regression
```bash
python src/logistic_regression.py
```

---

### Diagnostic distribution offset
```bash
python src/diagnose_shift.py
```

---

## 👥 Authors

*Qihui Ren & Group 12345: Jingyi Zhang,  Xu Han, Jingyun Li, Yanyan Zhu* · COMP9417 T1 2025

---

## version
> • **numpy**: 1.26.4
> 
> • **pandas**: 2.2.2
> 
> • **scikit-learn**: 1.5.1
> 
> • **xgboost**: 3.0.0
> 
> • **imbalanced-learn**: 0.12.3
> 
> • **tensorflow**: 2.19.0

---

## Parameter Description

### Model hyperparameters
The following parameters can be adjusted in the corresponding model file (such as xgboost. py):
```python
# XGBoost example
model = XGBClassifier(
    objective='multi:softprob',
    num_class=28,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    scale_pos_weight='balanced',  
    tree_method='gpu_hist',       
    ...
)

```

### Experimental configuration
- **use_smote** – Do you want to use SMOTE oversampling (True/False).
- **use_class_weight** - Whether to enable category weighting (True/False).
- **test_size** – Validation set ratio (default 0.2).

---

## output
- **After the model training is completed, the terminal will output the following indicators** –  
**Validation set** Weighted Logarithmic Loss, Accuracy, Weighted F1, Macro Average F1.
**Test Set 2** Weighted Logarithmic Loss, Accuracy, Weighted F1, Macro Average F1.
  
- **The distribution offset diagnosis results will generate label distribution maps and KS test statistical results.** –

- **Prediction files** - preds_1.npy, preds_2.npy
---

## precautions
- **GPU support** – If GPU acceleration is used (such as XGBoost's tree_sthod='gpu_ist '), CUDA and cuML libraries need to be installed.
- **Data consistency** – Ensure that the column names and formats of all CSV files are consistent with the expected code (300 dimensional features).
- **Result reproduction** – Set random_state=42 to ensure reproducibility.
