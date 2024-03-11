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

# Function to save training data with column names
def save_train_data(X_train_scaled, y_train, column_names, X_train_filepath, y_train_filepath):
    X_train_df = pd.DataFrame(X_train_scaled, columns=column_names)
    X_train_df.to_csv(X_train_filepath, index=False)
    y_train.to_csv(y_train_filepath, index=False)
    
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
def train_and_save_random_forest(X_train_scaled, y_train, filename='../models/random_forest_model.pkl', cv=3):
    # First, conduct a Randomized Search to narrow down the parameter space
    param_distributions = {'n_estimators': np.arange(100, 1001, 100), 
        'max_depth': [None] + list(np.arange(5, 51, 5)),  
        'min_samples_split': np.arange(2, 11, 1),  
        'min_samples_leaf': np.arange(1, 11, 1),  
        'bootstrap': [True, False]} 
    rf_model_random = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf_model_random,
        param_distributions= param_distributions,
        n_iter=8,
        cv=cv,
        random_state=42,
        n_jobs=-1)
    random_search.fit(X_train_scaled, y_train)
    best_rf_model = random_search.best_estimator_
    
    save_model(best_rf_model, filename)
    
    print("Best parameters from Randomized Search:", random_search.best_params_)
    print("Best score from Randomized Search:", random_search.best_score_)
    
    return best_rf_model

# Model 3 Training - k-Nearest Neighbor
def train_and_save_knn(X_train_scaled, y_train, filename='../models/knn_model.pkl'):
    model_knn = KNeighborsClassifier()
    model_knn.fit(X_train_scaled, y_train)
    save_model(model_knn, filename)
    return model_knn


from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Model 4 Training and Hyperparameter tuning - XGBoost
def train_and_save_xgboost(X_train_scaled, y_train, filename='../models/xgboost_model.pkl', cv=3, n_iter=8, random_state=42):
    param_dist = {'n_estimators': np.arange(50, 300, 50),
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': np.arange(3, 10, 2),
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]}
    
    xgb_clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=random_state)
    
    random_search.fit(X_train_scaled, y_train)
    best_xgb_model = random_search.best_estimator_
    
    save_model(best_xgb_model, filename)
    
    # Print the best parameters and score
    print("Best parameters from Randomized Search:", random_search.best_params_)
    print("Best score from Randomized Search:", random_search.best_score_)
    
    return best_xgb_model

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

# Main pipeline execution
def main():
    data_numeric = read_cleaned_data('../data/final/modeling_dataset_32k.csv')
    X, y, column_names = prepare_data_for_modeling(data_numeric, 'hit_song')
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    save_train_data(X_test_scaled, y_test, column_names,'../data/final/X_train_scaled.csv','../data/final/y_train.csv')
    save_test_data(X_test_scaled, y_test, column_names,'../data/final/X_test_scaled.csv','../data/final/y_test.csv')

    # Train and save each model
    models = {
        'logistic_regression': train_and_save_logistic_regression,
        'random_forest': train_and_save_random_forest,
        'knn': train_and_save_knn,
        'xgboost': train_and_save_xgboost}

    for name, train_function in models.items():
        print(f"Train and save {name} model")
        model = train_function(X_train_scaled, y_train)
        print("Plot Learning Curve:")
        print(plot_learning_curve(model, X_train_scaled, y_train, cv=3, scoring='accuracy', title='Learning Curve'))


if __name__ == '__main__':
    main()