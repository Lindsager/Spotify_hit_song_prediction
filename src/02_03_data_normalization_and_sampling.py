import pandas as pd
import numpy as np

# read-in merged hit songs dataset, filter out songs released prior to 1985
def process_billboard_hits_dataset(filepath):
    data = pd.read_csv(filepath)
    df_hits_cleaned = data[data['year'] >= 1985]
    
    return df_hits_cleaned

# read-in merged non-hit songs dataset, filter out songs before 1985, filter duration to match hit songs
def clean_non_hits_dataset(filepath, df_hits_cleaned):
    data = pd.read_csv(filepath)
    non_hit_filtered = data[data['year'] >= 1985]
    lower_bound = df_hits_cleaned['duration_ms'].describe()['min']
    upper_bound = df_hits_cleaned['duration_ms'].describe()['max']
    non_hit_cleaned = non_hit_filtered[(non_hit_filtered['duration_ms'] >= lower_bound) & 
                                       (non_hit_filtered['duration_ms'] <= upper_bound)]
    return non_hit_cleaned
    
# random stratified sampling of large dataset to ensure yearly distribution of sampled data reflects billboard distribuion
def sample_non_hits_clean_data(non_hit_cleaned, df_hits_cleaned):
    year_bins = [1984, 1994, 2004, 2014, 2023]
    labels = ['1985-1994', '1995-2004', '2005-2014', '2015-2023']
    
    df_hits_cleaned['year_bin'] = pd.cut(df_hits_cleaned['year'], bins=year_bins, labels=labels, right=True)
    hits_distribution = df_hits_cleaned['year_bin'].value_counts(normalize=True)
    non_hit_cleaned['year_bin'] = pd.cut(non_hit_cleaned['year'], bins=year_bins, labels=labels, right=True)
    
    non_hits_sampled = pd.DataFrame()
    
    total_non_hits_to_sample = len(df_hits_cleaned)
    
    for label in labels:
        num_to_sample = int(hits_distribution[label] * total_non_hits_to_sample)
        sampled = non_hit_cleaned[non_hit_cleaned['year_bin'] == label].sample(n=num_to_sample, random_state=42)
        non_hits_sampled = pd.concat([non_hits_sampled, sampled])
    
    non_hits_sampled.drop('year_bin', axis=1, inplace=True)
    df_hits_cleaned.drop('year_bin', axis=1, inplace=True)
    
    return non_hits_sampled

# Creation of combined dataset (16k hits + 16k non-hits) for modeling and exploratory data analysis
def create_modeling_dataset_32k(non_hits_sampled, df_hits_cleaned):
    modeling_dataset = pd.concat([df_hits_cleaned, non_hits_sampled], ignore_index=True)
    return modeling_dataset

#save the processed and final datasets for downstream model training and analysis
def save_processed_data(df, filepath):
    df.to_csv(filepath, index=False)

#save finalized hit and non-hit songs in processed, save final combined dataset in final
def main():
    df_hits_cleaned = process_billboard_hits_dataset('../data/processed/top_hits_features.csv')
    non_hit_cleaned = clean_non_hits_dataset('../data/processed/non_hit_tracks.csv', df_hits_cleaned)
    non_hits_sampled = sample_non_hits_clean_data(non_hit_cleaned, df_hits_cleaned)
    modeling_dataset = create_modeling_dataset_32k(non_hits_sampled, df_hits_cleaned)
    
    save_processed_data(df_hits_cleaned, '../data/processed/df_hits_cleaned.csv')
    save_processed_data(non_hits_sampled, '../data/processed/non_hits_sampled.csv')
    save_processed_data(modeling_dataset, '../data/final/modeling_dataset_32k.csv')

if __name__ == '__main__':
    main()