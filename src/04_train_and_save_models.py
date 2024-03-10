#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import joblib  # for saving models
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Read in the cleaned dataset and drop non-numeric features
def read_cleaned_data(filepath):
    data = pd.read_csv(filepath)
    data_numeric = data.drop(['song', 'artist', 'song_artist'], axis=1)
    return data_numeric

# Prepare features and target variable
def prepare_data_for_modeling(data_numeric, target_column):
    X = data_numeric.drop(target_column, axis=1)
    y = data_numeric[target_column]
    column_names = X.columns.tolist()
    return X, y, column_names

# Split the dataset into training and test sets
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Feature scaling
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Function to save test data with column names
def save_test_data(X_test_scaled, y_test, column_names, X_test_filepath, y_test_filepath):
    X_test_df = pd.DataFrame(X_test_scaled, columns=column_names)
    X_test_df.to_csv(X_test_filepath, index=False)
    y_test.to_csv(y_test_filepath, index=False)

# Function to save pre-trained models for easier testing 
def save_model(model, filename):
    joblib.dump(model, filename)

# Model 1 Training - Logistic Regression
def train_and_save_logistic_regression(X_train_scaled, y_train, filename='../models/logistic_regression_model.pkl'):
    model_lr = LogisticRegression()
    model_lr.fit(X_train_scaled, y_train)
    save_model(model_lr, filename)
    return model_lr

# Model 2 Training - Random Forest
def train_and_save_random_forest(X_train_scaled, y_train, filename='../models/random_forest_model.pkl'):
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)
    save_model(model_rf, filename)
    return model_rf

# Model 3 Training - k-Nearest Neighbor
def train_and_save_knn(X_train_scaled, y_train, filename='../models/knn_model.pkl'):
    model_knn = KNeighborsClassifier()
    model_knn.fit(X_train_scaled, y_train)
    save_model(model_knn, filename)
    return model_knn

# Model 4 Training - XGBoost
def train_and_save_xgboost(X_train_scaled, y_train, filename='../models/xgboost_model.pkl'):
    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(X_train_scaled, y_train)
    save_model(model_xgb, filename)
    return model_xgb

# Function for model evaluation
def evaluate_model(model, X_test_scaled, y_test):
    predictions = model.predict(X_test_scaled)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    # Note: precision_recall_curve returns 3 arrays: precision, recall, thresholds. They need to be handled if used.
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test_scaled)[:,1])
    # Implement logic to use or display precision-recall data as needed
    # For example, you might plot the precision-recall curve here
    
# Main pipeline execution
def main():
    # Assuming `data_filepath` and `target_column` are defined and the dataset is clean
    data_numeric = read_cleaned_data('../data/final/modeling_dataset_32k.csv')
    X, y, column_names = prepare_data_for_modeling(data_numeric, 'hit_song')
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    save_test_data(X_test_scaled, y_test, column_names,'../data/final/X_test_scaled.csv','../data/final/y_test.csv')

    # Train and save each model
    models = {
        'logistic_regression': train_and_save_logistic_regression,
        'random_forest': train_and_save_random_forest,
        'knn': train_and_save_knn,
        'xgboost': train_and_save_xgboost
    }

    for name, train_func in models.items():
        print(f"Train and save {name} model")
        model = train_func(X_train_scaled, y_train)
        evaluate_model(model, X_test_scaled, y_test)
        # Optionally, add a line here to plot or further analyze model performance

if __name__ == '__main__':
    main()


# In[ ]:




