import pandas as pd
import numpy as np
import joblib
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
def read_split_data(filepath_xtest, filepath_ytest):
    X_test_scaled = pd.read_csv(filepath_xtest)
    y_test = pd.read_csv(filepath_ytest).squeeze()
    return X_test_scaled, y_test

#load in the pre-trained models
def load_pretrained_models(lr_filepath, rf_filepath, knn_filepath, xgb_filepath):
    logistic_regression = joblib.load(lr_filepath)
    random_forest = joblib.load(rf_filepath)
    knn = joblib.load(knn_filepath)
    xgboost = joblib.load(xgb_filepath)
    return logistic_regression, random_forest, knn, xgboost

#print the confusion matrix in tabular form
def plot_confusion_matrix(y_test, predictions, model, labels=['Non-Hit', 'Hit']):
    conf_matrix = confusion_matrix(y_test, predictions)
    conf_matrix_df = pd.DataFrame(conf_matrix, 
                         index=['Actual Non-Hit Song', 'Actual Hit Song'], 
                         columns=['Predicted Non-Hit Song', 'Predicted Hit Song'])
    return conf_matrix_df

#print the feature importance (or coefficients) for each model
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

# format all of the output functions with text    
def evaluate_model(model, X_test_scaled, y_test, column_names):
    predictions = model.predict(X_test_scaled)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix Results:")
    print(plot_confusion_matrix(y_test, predictions, model, labels=['Non-Hit', 'Hit']))
    print('Feature Importance Results:')
    print(print_feature_importance(model, column_names))  

    
# Main pipeline execution
def main():
    X_test_scaled, y_test = read_split_data('../data/final/X_test_scaled.csv', 
                                            '../data/final/y_test.csv' )
    
    logistic_regression, random_forest, knn, xgboost = load_pretrained_models('../models/logistic_regression_model.pkl',
                                                                              '../models/random_forest_model.pkl', 
                                                                              '../models/knn_model.pkl',
                                                                              '../models/xgboost_model.pkl')
    column_names = (pd.read_csv('../data/final/X_test_scaled.csv')).columns.tolist()
    models = [logistic_regression, random_forest, knn, xgboost]
    for model in models:
            print("____________________________________________________________________________________________")
            print("____________________________________________________________________________________________")
            print(f"Evaluating {model} model...")
            evaluate_model(model, X_test_scaled, y_test, column_names)



if __name__ == '__main__':
    main()
