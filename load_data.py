import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def load_data(base_path = 'D:/p_code/unsw/comp9417/proj/Group Project - Data-20250401'):
    # Define 300 feature names
    feature_columns = [f'f{i}' for i in range(300)]
    
    # Load the training data and labels and skip the first row
    X_train = pd.read_csv(os.path.join(base_path, 'X_train.csv'),
                          skiprows=1, header=None, names=feature_columns)
    y_train = pd.read_csv(os.path.join(base_path, 'y_train.csv'),
                          skiprows=1, header=None).squeeze()
    
    # Load the test dataset and skip the first row
    X_test1 = pd.read_csv(os.path.join(base_path, 'X_test_1.csv'),
                          skiprows=1, header=None, names=feature_columns)
    X_test2 = pd.read_csv(os.path.join(base_path, 'X_test_2.csv'),
                          skiprows=1, header=None, names=feature_columns)
    
    # For y_test2, since only the first 202 lines have labels, take the first 202 lines and convert them into an array
    y_test2 = pd.read_csv(os.path.join(base_path, 'y_test_2_reduced.csv'),
                          skiprows=1, header=None).squeeze()[:202].values

    # Data preprocessing
    # Fill in the missing values in the training set using the median in the training set
    X_train = X_train.fillna(X_train.median())
    y_train = y_train.fillna(-1).astype(int)

    # The test set is also filled with the median of the training set to ensure data consistency
    X_test1 = X_test1.fillna(X_train.median())
    X_test2 = X_test2.fillna(X_train.median())
    
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test1 = scaler.transform(X_test1)
    X_test2 = scaler.transform(X_test2)
    
    return X_train, y_train, X_test1, X_test2, y_test2