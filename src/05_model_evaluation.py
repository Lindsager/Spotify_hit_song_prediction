#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import joblib  # for saving models
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Read in test data
def read_split_data(filepath_xtrain, filepath_xtest, filepath_ytrain, filepath_ytest):
    X_train_scaled = pd.read_csv(filepath_xtrain)
    X_test_scaled = pd.read_csv(filepath_xtest)
    y_train = pd.read_csv(filepath_ytrain).squeeze()
    y_test = pd.read_csv(filepath_ytest).squeeze()
    return X_train_scaled, X_test_scaled, y_train, y_test

def load_pretrained_models(lr_filepath, rf_filepath, knn_filepath, xgb_filepath)
    logistic_regression = joblib.load(lr_filepath)
    random_forest = jolib.load(rf_filepath)
    knn = jolib.load(knn_filepath)
    xgboost = jolib.load(xgboost_filepath)
    return logistic_regression, random_forest, knn, xgboost, model

def plot_confusion_matrix(y_test, predictions, model, labels=['Non-Hit', 'Hit']):
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_precision_recall_curve(y_test, predictions, title='Precision-Recall Curve'):
    precision, recall, thresholds = precision_recall_curve(y_test, predictions)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(5, 3))
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}', linewidth=2)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_learning_curve(model, X_train_scaled, y_train, cv=3, scoring='accuracy', title='Learning Curve'):
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_scaled, y_train, train_sizes=train_sizes, cv=cv, scoring=scoring
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.15)
    
    plt.plot(train_sizes, val_mean, label='Cross-validation score', color='green', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='green', alpha=0.15)
    
    plt.title(title)
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    
def print_feature_importance(model, column_names):
    # Logistic regression has coefficients
    if hasattr(model, 'coef_'):
        print("Logistic Regression - Using coef_ for feature importance.")
        importance = model.coef_[0] / np.sum(np.abs(model.coef_[0]))
        feature_importance_df = pd.DataFrame({
            'Feature': column_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        print(feature_importance_df)
    # RandomForest and XGBoost have feature importance
    elif hasattr(model, 'feature_importances_'):
        print("Tree-based Model - Using feature_importances_ for feature importance.")
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': column_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        print(feature_importance_df)
    # KNN doesn't have extractable feature importance
    else:
        print(f"{type(model).__name__} does not provide feature importance.")

    
def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, column_names):
    predictions = model.predict(X_test_scaled)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix Results:")
    print(plot_confusion_matrix(y_test, predictions, model, labels=['Non-Hit', 'Hit']))
    print("Precision-Recall Curve:")
    print(plot_precision_recall_curve(y_test, predictions, title='Precision-Recall Curve'))
    print("Plot Learning Curve:")
    print(plot_learning_curve(model, X_train_scaled, y_train, cv=3, scoring='accuracy', title='Learning Curve'))
    print('Feature Importance Results:')
    print(print_feature_importance(model, column_names))  

    
# Main pipeline execution
def main():


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
        evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, column_names)
        # Optionally, add a line here to plot or further analyze model performance

if __name__ == '__main__':
    main()
#Train and save each model as .pkl, save test data (X_test_scaled, y_test)
def main():
    x_train_scaled, X_test_scaled, y_train, y_test = read_split_data('../data/final/X_train_scaled.csv', '../data/final/X_test_scaled.csv', '../data/final/y_test.csv', '../data/final/y_test.csv' )
    
    logistic_regression, random_forest, knn, xgboost = import_pretrained_models('../models/logistic_regression_model.pkl', '../models/random_forest_model.pkl', '../models/knn_model.pkl', '../models/xgboost_model.pkl')
    
    
    



if __name__ == '__main__':
    main()