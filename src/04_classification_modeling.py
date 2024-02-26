#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# ## Read in intermediate datasets from 03_Exploratory_Data_Analysis

# In[3]:


# Consolidated hit and non-hit songs 
data = pd.read_csv('/Users/tiffanytong/Documents/GitHub/1/data/final/modeling_dataset_32k.csv')


# In[4]:


# Drop non-numeric columns
data_numeric = data.drop(['song', 'artist', 'song_artist'], axis=1)


# # 7. Modeling

# In[5]:


# Split data into features (X) and labels (y)
X = data_numeric.drop('hit_song', axis=1)
y = data_numeric['hit_song']


# In[6]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Logistic Regression model 

# ### 1) Logistic Regression model initiation and training using all features in dataset

# In[8]:


model_lr = LogisticRegression()
model_lr.fit(X_train_scaled, y_train)


# ### 2) Make prediction using the Logistic Regression model trained with all features

# In[9]:


y_pred_lr = model_lr.predict(X_test_scaled)


# In[10]:


# Evaluate the model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_lr


# In[11]:


class_report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
pd.DataFrame(class_report_lr).transpose()


# In[12]:


conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Hit', 'Hit'], yticklabels=['Low Popularity', 'High Popularity'])
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[13]:


from sklearn.metrics import precision_recall_curve, auc

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_lr)
pr_auc = auc(recall, precision)

# Plot Precision-Recall Curve
plt.figure(figsize=(5, 3))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Logistic Regression')
plt.legend()
plt.show()


# ### 3) Extract the coefficients of the features
# #### *Identify the features with the most significant relationship with the likelihood of a song being a hit or non-hit.*

# In[14]:


# Extracting the coefficients
coefficients = model_lr.coef_[0]
features = X.columns


# In[15]:


# Create a DataFrame to display feature names and their corresponding coefficients
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})

# Sort the coefficients for better visualization
coef_df['Absolute_Coefficient'] = coef_df['Coefficient'].abs()
coef_df_abs = coef_df.sort_values(by='Absolute_Coefficient', ascending=False)
coef_df_abs


# ### 4) Train another Logistic Regression model with the top 5 features and make prediction

# In[16]:


features_to_use = ['instrumentalness', 'loudness', 'energy', 'acousticness', 'danceability']
X_5f = data_numeric[features_to_use]
y_5f = data_numeric['hit_song']
X_train_5f, X_test_5f, y_train_5f, y_test_5f = train_test_split(X_5f, y_5f, test_size=0.2, random_state=42)


# In[17]:


scaler = StandardScaler()
X_train_5f = scaler.fit_transform(X_train_5f)
X_test_5f = scaler.transform(X_test_5f)


# In[18]:


model_lr_5f = LogisticRegression()
model_lr_5f.fit(X_train_5f, y_train_5f)


# In[19]:


y_pred_5f = model_lr_5f.predict(X_test_5f)


# In[20]:


# Evaluate the model
accuracy_lr_5f = accuracy_score(y_test_5f, y_pred_5f)
accuracy_lr_5f


# In[21]:


class_report_lr_5f = classification_report(y_test_5f, y_pred_5f, output_dict=True)
pd.DataFrame(class_report_lr_5f).transpose()


# In[22]:


conf_matrix_lr_5f = confusion_matrix(y_test_5f, y_pred_5f)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix_lr_5f, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Hit', 'Hit'], yticklabels=['Low Popularity', 'High Popularity'])
plt.title('Logistic Regression Confusion Matrix - 5 Features')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# ## Random Forest Classifier model 

# ### 1) Random Forest Classifier model initiation and training

# In[23]:


# Initialize and train the RandomForestClassifier with all features
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)


# In[24]:


# Make prediction
y_pred_rf = rf_model.predict(X_test_scaled)


# In[25]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_rf)
accuracy


# In[26]:


class_report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
pd.DataFrame(class_report_rf).transpose()


# In[27]:


conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Popularity', 'High Popularity'], yticklabels=['Low Popularity', 'High Popularity'])
plt.title('Random Forest Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# ### 2) Plot a single decision tree from the Random Forest, with the first three levels of dept

# In[28]:


from sklearn.tree import plot_tree

# Select one tree from the random forest
single_tree = rf_model.estimators_[0]

# Set the figure size for the plot
plt.figure(figsize=(20,10))

# Plot the tree
plot_tree(single_tree, 
          filled=True, 
          feature_names=X_train.columns, 
          class_names=['Not Popular', 'Popular'], 
          max_depth=3,  # Set the maximum depth to avoid a very large tree
          fontsize=10)

# Show the plot
plt.show()


# ### 3) Feature importance in this RandomForestClassifier model

# In[29]:


feature_importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)


# ### 4) Hyperparameter Tuning the Random Forest Classifier
# #### *Use Randomized Search with Cross-Validation to identify the best hyperparameter values for the Random Forest classifier.*

# In[30]:


# Fine-tune the Random Forest model using Randomized Search to improve its accuracy
from sklearn.model_selection import RandomizedSearchCV

# Hyperparameter grid to sample from during fitting
param_distributions = {'n_estimators': np.arange(100, 1001, 100), 
    'max_depth': [None] + list(np.arange(5, 51, 5)),  
    'min_samples_split': np.arange(2, 11, 1),  
    'min_samples_leaf': np.arange(1, 11, 1),  
    'bootstrap': [True, False]}  


# In[31]:


# Initialize another RandomForestClassifier
rf_model_tuned = RandomForestClassifier(random_state=42)

# Initialize RandomizedSearchCV with 3-fold cross-validation
random_search = RandomizedSearchCV(estimator=rf_model_tuned, param_distributions=param_distributions, n_iter=8,
    cv=3, verbose=2, random_state=42, n_jobs=-1)


# In[32]:


# Start Timing
import time
start_time = time.time()


# In[33]:


# Fit RandomizedSearchCV
random_search.fit(X_train_scaled, y_train)


# In[34]:


# End timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for RandomizedSearchCV fitting: {elapsed_time} seconds")


# In[35]:


# Get the best parameters from the randomized search
print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)


# In[36]:


# Use the best model from randomized search
best_rf_model = random_search.best_estimator_
best_y_pred_rf = best_rf_model.predict(X_test_scaled)


# In[37]:


# Plot a single decision tree
from sklearn.tree import plot_tree

# Select one tree from the random forest
single_tree = best_rf_model.estimators_[0]

# Set the figure size for the plot
plt.figure(figsize=(20,10))

# Plot the tree
plot_tree(single_tree, 
          filled=True, 
          feature_names=X_train.columns, 
          class_names=['Not Popular', 'Popular'], 
          max_depth=3,  # Set the maximum depth to avoid a very large tree
          fontsize=10)

# Show the plot
plt.show()


# In[38]:


# Evaluate the model
accuracy_rf_best = accuracy_score(y_test, best_y_pred_rf)
accuracy_rf_best


# In[39]:


conf_matrix_kn = confusion_matrix(y_test, best_y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix_kn, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Popularity', 'High Popularity'], yticklabels=['Low Popularity', 'High Popularity'])
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[40]:


class_report_rf_best = classification_report(y_test, best_y_pred_rf, output_dict=True)
pd.DataFrame(class_report_rf_best).transpose()


# In[41]:


# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, best_y_pred_rf)
pr_auc = auc(recall, precision)

# Plot Precision-Recall Curve
plt.figure(figsize=(5, 3))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Random Forest')
plt.legend()
plt.show()


# ## K-Nearest Neighbors Classifier model

# ### 1) K-Nearest Neighbors Classifier model initiation and training

# In[43]:


# Initialize and train the KNeighborsClassifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)


# In[44]:


y_pred_kn = knn_model.predict(X_test_scaled)


# In[45]:


# Evaluate the model
accuracy_kn = accuracy_score(y_test, y_pred_kn)
accuracy_kn


# In[46]:


class_report_kn = classification_report(y_test, y_pred_kn, output_dict=True)
pd.DataFrame(class_report_kn).transpose()


# In[47]:


conf_matrix_kn = confusion_matrix(y_test, y_pred_kn)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix_kn, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Popularity', 'High Popularity'], yticklabels=['Low Popularity', 'High Popularity'])
plt.title('KNeighbors Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[48]:


from sklearn.metrics import precision_recall_curve, auc

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_kn)
pr_auc = auc(recall, precision)

# Plot Precision-Recall Curve
plt.figure(figsize=(5, 3))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - KNN')
plt.legend()
plt.show()


# ### 2) K-Nearest Neighbors 3D Clustering

# In[49]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Selecting features
instrumentalness, danceability, energy = X_train_scaled[:, 7], X_train_scaled[:, 0], X_train_scaled[:, 1]

fig = plt.figure(figsize=(12, 6)) 
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for non-hit and hit songs
ax.scatter(instrumentalness[y_train == 0], danceability[y_train == 0], energy[y_train == 0],
           c='blue', marker='o', label='Non-hit') 
ax.scatter(instrumentalness[y_train == 1], danceability[y_train == 1], energy[y_train == 1],
           c='red', marker='^', label='Hit')  

ax.set_xlabel('Instrumentalness')
ax.set_ylabel('Danceability')
ax.set_zlabel('Energy')
ax.set_title('3D Clustering of Hit and Non-Hit Songs - 3 Features')  
ax.legend()


# ## XGBoost Classifier model

# ### 1) XGBoost Classifier model initiation and training

# In[50]:


xgb_clf = xgb.XGBClassifier(eval_metric='logloss')
xgb_clf.fit(X_train_scaled, y_train)


# In[51]:


# Prediction
y_pred_xgb = xgb_clf.predict(X_test_scaled)


# In[52]:


# Evaluate the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
accuracy_xgb


# In[53]:


class_report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
pd.DataFrame(class_report_xgb).transpose()


# In[54]:


conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Popularity', 'High Popularity'], yticklabels=['Low Popularity', 'High Popularity'])
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[55]:


# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_xgb)
pr_auc = auc(recall, precision)

# Plot Precision-Recall Curve
plt.figure(figsize=(5, 3))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - XGBoost')
plt.legend()
plt.show()


# ### 2) Hyperparameter Tuning the XGBoost Classifier

# In[56]:


# Define the parameter grid for RandomizedSearchCV
param_dist = {'n_estimators': np.arange(50, 300, 50),
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': np.arange(3, 10, 2),
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]}


# In[57]:


# Initialize the XGBoost classifier
xgb_clf_tuned = xgb.XGBClassifier(eval_metric='logloss')


# In[58]:


from sklearn.model_selection import RandomizedSearchCV

# Initialize RandomizedSearchCV with 3-fold cross-validation
random_search = RandomizedSearchCV(xgb_clf, param_distributions=param_dist, n_iter=8, cv=3, verbose=1, n_jobs=-1, random_state=42)


# In[59]:


import time

# Start timing
start_time = time.time()


# In[60]:


# Fit RandomizedSearchCV
random_search.fit(X_train_scaled, y_train)


# In[61]:


# End timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for RandomizedSearchCV fitting with XGBoost: {elapsed_time} seconds")


# In[62]:


# Best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)


# In[63]:


# Predict using the best estimator
best_xgb_clf = random_search.best_estimator_
y_pred_xgb_best = best_xgb_clf.predict(X_test_scaled)


# In[64]:


# Evaluate the model
accuracy_xgb_best = accuracy_score(y_test, y_pred_xgb_best)
accuracy_xgb_best


# In[65]:


conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb_best)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Popularity', 'High Popularity'], yticklabels=['Low Popularity', 'High Popularity'])
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[66]:


class_report_xgb_best = classification_report(y_test, y_pred_xgb_best, output_dict=True)
pd.DataFrame(class_report_xgb_best).transpose()


# In[67]:


# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_xgb_best)
pr_auc = auc(recall, precision)

# Plot Precision-Recall Curve
plt.figure(figsize=(5, 3))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - XGBoost')
plt.legend()
plt.show()


# ### 3) Plot the learning curve of the XGBoost model that has been optimized

# In[68]:


from sklearn.model_selection import learning_curve

train_sizes = np.linspace(0.1, 1.0, 5)

# Compute the learning curve
train_sizes, train_scores, val_scores = learning_curve(
    best_xgb_clf, X_train_scaled, y_train,
    train_sizes=train_sizes, cv=3, scoring='accuracy')

# Calculate the average and standard deviation of the training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plotting the learning curve
plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.15)

plt.plot(train_sizes, val_mean, label='Cross-validation score', color='green', marker='o')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='green', alpha=0.15)

plt.title('Learning Curve for XGBoost Model')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.legend()

plt.show()


# #### *The training score and the cross-validation score have a trend to converge if more training data is available, which indicates the model may generalize better with larger datasets.*

# ### 3) Check the feature importance of the hyperparameter tuned XGBoost Classifier

# In[69]:


# Check feature importance
importances = best_xgb_clf.feature_importances_


# In[70]:


feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance DataFrame
print(feature_importance_df)

