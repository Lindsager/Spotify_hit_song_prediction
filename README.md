Hit Song Predicton by Audio Features and Release Year
Project Overview:
This project aims to predict whether a song will be a hit on the Billboard Top 100 chart based on its audio features and release year. Utilizing data from the Billboard Top 100 songs and 2.3 million Spotify songs, we have built a machine learning model that classifies songs as hits or non-hits and identified the most important audio features for predicting Billboard Top 100 hit songs.

Data Sources:
The  dataset composed of audio features from hit and non-hit Spotify songs across 4 Kaggle datasets:
 - Source of truth for Billboard Top 100: Billboard "The Hot 100" Songs (https://www.kaggle.com/datasets/dhruvildave/billboard-the-hot-100-songs)
 - Billboard data w/ audio features: Billboard Hot weekly charts (https://www.kaggle.com/datasets/thedevastator/billboard-hot-100-audio-features)
 - 1.2M songs w/ audio features: Spotify 1.2M+ Songs (https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)
 - 1M songs w/ audio features: Spotify_1Million_Tracks (https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks)

Final Dataset:
 - The final modeling dataset is composed of 16,242 Billboard Top 100 songs with audio features and 16,242 non-hit songs with audio features, all from the period of 1985-2023

Model Features:
The final XGBoost model considers various audio features, including
- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Valence
- Release Year




