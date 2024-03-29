# Hit Song Predicton by Audio Features and Release Year
## Project Overview:
This project aims to predict whether a song will be a hit on the Billboard Top 100 chart based on its audio features and release year. Utilizing data from the Billboard Top 100 songs and 2.3 million Spotify songs, we have built a machine learning model that classifies songs as hits or non-hits and identified the most important audio features for predicting Billboard Top 100 hit songs.

## Data Sources:
The  dataset composed of audio features from hit and non-hit Spotify songs across 4 Kaggle datasets:
 - Source of truth for Billboard Top 100: Billboard "The Hot 100" Songs (https://www.kaggle.com/datasets/dhruvildave/billboard-the-hot-100-songs)
 - Billboard data w/ audio features: Billboard Hot weekly charts (https://www.kaggle.com/datasets/thedevastator/billboard-hot-100-audio-features)
 - 1.2M songs w/ audio features: Spotify 1.2M+ Songs (https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)
 - 1M songs w/ audio features: Spotify_1Million_Tracks (https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks)

## Final Dataset:
 - The final modeling dataset is composed of 16,242 Billboard Top 100 songs with audio features and 16,242 non-hit songs with audio features, all from the period of 1985-2023

## Model Features:
The final XGBoost model considers various audio features, including

![image](https://github.com/Lindsager/Spotify_hit_song_prediction/assets/102261851/f77c877c-bf24-4a59-b7df-8ee926ffa01c)

## Notebooks (Includes data processing, exploratory data analysis and modeling process with visual components)
 - 01_data_preparation_and_merging.ipynb: this reads in all raw data files, normalizes formatting across datasets, performs merging and removes overlapping samples [output = processed datasets]
 - 02_data_normalization_and_sampling.ipynb: this reads in the pre-processed data files, handles outliers, sets minimum sampling year to 1985, bins non-hit samples by year and performs random stratified sampling on non-hit songs [output = processed datasets]
 - 03_Exploratory_Data_Analysis.ipynb: this contains all visualizations for distribution and time-series analyses as well as feature correlation [output = final dataset]
 - 04_classification_modeling.ipynb: this trains 4 models (logistic regression, random forest, k-nearest neighbors and xgboost) and completes hyper parameter tuning by random search for tree-based models. visualizations of model evaluation are included [output = model evaluation]
 
## Pipeline (From data processing through model evaluation)
- 01_data_pre_processing.py: this reads in all raw data files, normalizes formatting across datasets, performs merging and removes overlapping samples [output = processed datasets]
- 02_03_data_normalization_and_sampling.py: this reads in the pre-processed data files, handles outliers, sets minimum sampling year to 1985, bins non-hit samples by year and performs random stratified sampling on non-hit songs [output = final dataset]
- 04_train_and_save_models.py: this trains 4 models (logistic regression, random forest, k-nearest neighbors and xgboost), completes hyper parameter tuning by random search for tree-based models and saves the best model of each as a .pkl file [output = pre-trained models]
- 05_model_evaluation.py: this evaluates each of the pre-trained models and returns success metrics such as accuracy and precision-recall

## Running the Model Evaluation Script (Recommendation - Run from a virtual environment):
1. Clone the repository: https://github.com/Lindsager/Spotify_hit_song_prediction
2. From command line, navigate to repository directory location (folder level = src): (MacOS) cd Users/Documents/GitHub/Spotify_hit_song_prediction/src, (Windows) cd "Users\Documents\GitHub\Spotify_hit_song_prediction\src" 
3. Create the virtual environment:
   - macOS/Linux
     - python3 -m venv venv
     - source venv/bin/activate
   - Windows
     - python -m venv venv
     - .\venv\Scripts\activate
4. Prerequisites:
   - Install requirements.txt:
     - MacOS: pip install -r ../requirements.txt
     - Windows: pip install -r ..\requirements.txt
5. Run pre-trained model evaluation script: python 05_model_evaluation.py
6. The output should include tabular confusion matrix, accuracy, precision, recall F1 scores and feature/coefficient importance for each of the models [logistic regression, random forest, knn, XGBoost]

## Running the Entire Data Science Pipeline (Recommendation - Run from a virtual environment):
1. Clone the repository: https://github.com/Lindsager/Spotify_hit_song_prediction
2. From command line, navigate to repository directory location (folder level = src): similar to --> (MacOS) cd Users/Documents/GitHub/Spotify_hit_song_prediction/src, (Windows) cd "Users\Documents\GitHub\Spotify_hit_song_prediction\src" 
3. Create the virtual environment:
   - macOS/Linux
     - python3 -m venv venv
     - source venv/bin/activate
   - Windows
     - python -m venv venv
     - .\venv\Scripts\activate
4. Prerequisites:
   - Install requirements.txt:
     - MacOS: pip install -r ../requirements.txt
     - Windows: pip install -r ..\requirements.txt
5. Run the data processing pipeline:
   - python 01_data_pre_processing.py
     - expected output: semi-processed hit and non-hit song datasets
   - python 02_03_data_normalization_and_sampling.py
     - expected output: finalized and normalized combined hit/non-hit song dataset for model training, separate finalized datasets for hit and non-hit songs
6. Run the model training, tuning and evaluation pipeline:
   - python 04_train_and_save_models.py
     - expected output: pre-trained and tuned classification models saved as .pkl files
   - python 05_model_evaluation.py
     - expected output: tabular confusion matrix, accuracy, precision, recall F1 scores and feature/coefficient importance for each of the models 

