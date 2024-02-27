#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import plot_importance
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read in the cleaned dataset
def read_cleaned_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Prepare features and target variable
def prepare_data_for_modeling(df, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y

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


# In[ ]:


# Train the logistic regression model
def train_logistic_regression(X_train_scaled, y_train):
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    return model

# Evaluate the logistic regression model
def evaluate_model(model, X_test_scaled, y_test):
    predictions = model.predict(X_test_scaled)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(precision-recall_curve(y_test, predictions))

# Extract the coefficients from the logistic regression model
def summarize_coefficients(model, feature_names):
    feature_importance = pd.DataFrame({'feature': feature_names, 'coefficient': model.coef_[0]})
    feature_importance['odds_ratio'] = np.exp(feature_importance['coefficient'])
    feature_importance = feature_importance.sort_values(by='coefficient', ascending=False)
    return feature_importance

# Train the logistic regression model
def train_logistic_regression(X_train_scaled, y_train):
    model_lr = LogisticRegression()
    model_lr.fit(X_train_scaled, y_train)
    return model

# Train another Logistic Regression model with the top 5 features and evaluate
def prepare_data(data, top_5_features, target, test_size=0.2, random_state=42):
    X = data[top_5_features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_logistic_regression(X_train_scaled, y_train):
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_model(model, X_test_scaled, y_test):
    predictions = model.predict(X_test_scaled)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))


# In[ ]:


# Train the Random Forest Classifier model and evaluate
def train_random_forest(X_train_scaled, y_train, n_estimators=100, max_depth=None, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model

def evaluate_model(model, X_test_scaled, y_test):
    predictions = model.predict(X_test_scaled)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(precision-recall_curve(y_test, predictions))

# Plot a single decision tree from the Random Forest
def plot_single_decision_tree(model, feature_names, class_names, max_depth):
    single_tree = model.estimators_[0]
    plot_tree()
    plt.show()

# Feature importance in this RandomForestClassifier model
def print_feature_importances(model, features):
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)

# Hyperparameter Tuning the Random Forest Classifier
def hyperparameter_tuning_rf(X_train_scaled, y_train, cv=3, n_iter=8, random_state=42):
    param_distributions = {}
    rf_model = RandomForestClassifier(random_state=random_state)
    random_search = RandomizedSearchCV()
    random_search.fit(X_train_scaled, y_train)
    return random_search.best_estimator_

# Evaluate the optimized Random Forest model
def evaluate_model(best_rf_model, X_test_scaled, y_test):
    best_rf_model = hyperparameter_tuning_rf(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(precision-recall_curve(y_test, predictions))


# In[ ]:


# Train the K-Nearest Neighbors Classifier model and evaluate
def train_knn(X_train_scaled, y_train):
    model = KNeighborsClassifier()
    return model

def evaluate_model(model, X_test_scaled, y_test):
    predictions = model.predict(X_test_scaled)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(precision-recall_curve(y_test, predictions))

# Plot K-Nearest Neighbors 3D Clustering
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_clustering(X_train_scaled, y_train, feature_indices):
    feature1, feature2, feature3 = X_train_scaled[:, feature_indices[0]], X_train_scaled[:, feature_indices[1]], X_train_scaled[:, feature_indices[2]]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feature1[y_train == 0], feature2[y_train == 0], feature3[y_train == 0])
    ax.scatter(feature1[y_train == 1], feature2[y_train == 1], feature3[y_train == 1])
    plt.show()


# In[ ]:


# Train the XGBoost Classifier model and evaluate
def train_xgb(X_train_scaled, y_train):
    model = xgb.XGBClassifier()
    return model

def evaluate_model(model, X_test_scaled, y_test):
    predictions = model.predict(X_test_scaled)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(precision-recall_curve(y_test, predictions))

# Hyperparameter Tuning the XGBoost Classifier
def hyperparameter_tuning_xgb(X_train_scaled, y_train, cv=3, n_iter=8, random_state=42):
    xgb_clf_tuned = xgb.XGBClassifier(eval_metric='logloss')
    random_search = RandomizedSearchCV()
    random_search.fit(X_train_scaled, y_train)
    return random_search.best_estimator_

# Evaluate the optimized XGBoost Classifier
def evaluate_model(best_rf_model, X_test_scaled, y_test):
    best_rf_model = hyperparameter_tuning_xgb(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(precision-recall_curve(y_test, predictions))

# Plot the learning curve of the optimized XGBoost Classifier
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X_train_scaled, y_train, cv=3, scoring='accuracy'):
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_scaled, y_train, train_sizes=train_sizes, cv=cv, scoring=scoring)
    plt.title('Learning Curve for XGBoost Model')
    plt.show()

# Feature importance in the optimized XGBoost Classifier
def print_feature_importances(best_rf_model, features):
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)

