#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
from datetime import datetime

# read-in data sources and pre-process by filtering columns, removing punctuation/capitalization, date formatting
# normalize song/artist fields, creating universal song_artist field for merging
def pre_process_billboard_source(filepath):
    data = pd.read_csv(filepath)
    data['year'] = pd.to_datetime(data['Date'], errors='coerce').dt.year
    data = data[['Song', 'Artist', 'year']]
    data.rename(columns={'Song': 'song', 'Artist': 'artist'}, inplace= True)
    data['song'] = data['song'].str.lower().str.strip()
    data['artist'] = data['artist'].str.lower().str.strip()
    data['song_artist'] = data['song'] + "_" + data['artist']

    return data.drop_duplicates(subset=['song','artist'])


def pre_process_billboard_with_features(filepath):
    data = pd.read_csv(filepath)
    data = data[['Performer', 'Song', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]
    data = data.dropna()
    data.rename(columns={'Song':'song', 'Performer':'artist'}, inplace=True)
    data['song'] = data['song'].str.lower().str.strip()
    data['artist'] = data['artist'].str.lower().str.strip()
    data['song_artist'] = data['song'] + "_" + data['artist']

    return data.drop_duplicates(subset=['song','artist'])

def pre_process_spotify_mil_2000_2023(filepath):
    df = pd.read_csv(filepath)
    df.rename(columns={'track_name': 'song', 'artist_name': 'artist'}, inplace=True)
    df['song'] = df['song'].str.lower().str.strip()
    df['artist'] = df['artist'].str.lower().str.strip()
    df['song_artist'] = df['song'] + "_" + df['artist']

    return df.drop_duplicates(subset=['song','artist'])


def pre_processs_spotify_1mil(filepath):
    df = pd.read_csv(filepath)
    correction_year = (df['year'] == 0) & (df['album'] == 'Optimism 2') & (df['artists'].str.contains('icizzle'))
    df.loc[correction_year, 'year'] = 2018
    df.rename(columns={'name': 'song', 'artists': 'artist'}, inplace=True)
    df['artist'] = df['artist'].str.replace(r"[\[\]()']", '', regex=True)
    df['song'] = df['song'].str.lower().str.strip()
    df['artist'] = df['artist'].str.lower().str.strip()
    df['song_artist'] = df['song'] + "_" + df['artist']

    return df.drop_duplicates(subset=['song','artist'])

# merge the non-hit song datasets and drop duplicate songs
def merge_1mil_datasets(df1, df2):
    conserved_columns = list(df1.columns.intersection(df2.columns))
    merged_df = pd.merge(df1[conserved_columns], df2[conserved_columns], on=conserved_columns, how='outer')
    merged_df_unique = merged_df.drop_duplicates(subset=['song_artist'], keep='first')

    return merged_df_unique


#merge billboard top songs datasets, assign audio features to our source of truth from the other, less complet dataset
def merge_hit_song_datasets(data_hits, billboard_hits_with_features):
    common_columns_hits = ['song', 'artist', 'song_artist']
    top_hits_all = pd.merge(data_hits, billboard_hits_with_features, on=common_columns_hits, how='left')
    return top_hits_all


# for any hit songs that don't have acssociated audio feature values, see if 2 mil dataset contains those songs
# merge hit song data from 2mil song dataset into hit songs dataset and then remove hit song data from 2mil dataset all together
def create_and_label_hit_and_nonhit_datasets(df_2mil, billboard_hits):
    overlap_tracks = df_2mil[df_2mil['song_artist'].isin(billboard_hits['song_artist'])]
    common_columns = list(billboard_hits.columns.intersection(overlap_tracks.columns))

    merged_hits_features = pd.merge(overlap_tracks, billboard_hits, on=common_columns, how='outer')
    merged_hits_features = merged_hits_features.drop_duplicates(subset='song_artist', keep='first')
    merged_hits_features = merged_hits_features[[col for col in merged_hits_features.columns if not col.endswith('_y')]]
    merged_hits_features = merged_hits_features.dropna()

    #label new column of resulting dataframes with 1=hit song and 0=non-hit song, this will be our prediction column
    merged_hits_features.loc[:,'hit_song'] = 1

    # create non-hit songs dataset by excluding overlaps from the hit songs dataset
    non_hit_songs = df_2mil[~df_2mil['song_artist'].isin(billboard_hits['song_artist'])]
    non_hit_songs.loc[:,'hit_song'] = 0

    return merged_hits_features, non_hit_songs

#save the intermediate outputs for use in the next portion of the pipeline
def save_processed_data(df, filepath):
    df.to_csv(filepath, index=False)


def main():
    data_hits = pre_process_billboard_source('../data/raw/charts_billboard_1958_2024.csv')
    df_hits = pre_process_billboard_with_features('../data/raw/hot_100_with_audio_features.csv')
    spotify_mil_2000_2023 = pre_process_spotify_mil_2000_2023('../data/raw/spotify_data.csv')
    spotify_1mil = pre_processs_spotify_1mil('../data/raw/spotify_1million.csv')

    top_hits_all = merge_hit_song_datasets(data_hits, df_hits)

    merged_spotify_datasets = merge_1mil_datasets(spotify_1mil, spotify_mil_2000_2023)

    hits_features, non_hits = create_and_label_hit_and_nonhit_datasets(merged_spotify_datasets, top_hits_all)

    save_processed_data(hits_features, '../data/processed/top_hits_features.csv')
    save_processed_data(non_hits, '../data/processed/non_hit_tracks.csv')

if __name__ == '__main__':
    main()