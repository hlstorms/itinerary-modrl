import pandas as pd
import numpy as np
from itertools import product

# Load and sort data by user and timestamp
df = pd.read_csv("data/train_data.csv", usecols=["id_veronacard", "ora_visita", "poi"])
df = df.sort_values(by=['id_veronacard', 'ora_visita'])

# Create the 'next_attraction' column by shifting within each user group
df['next_attraction'] = df.groupby('id_veronacard')['poi'].shift(-1)

# Drop rows where 'next_attraction' is NaN (end of sequence for each user)
df = df.dropna(subset=['next_attraction'])

# Drop duplicate transitions per user to count each unique pair once per user
df_unique_transitions = df.drop_duplicates(subset=['id_veronacard', 'poi', 'next_attraction'])

# Count transitions between each pair of attractions
transition_counts = df_unique_transitions.groupby(['poi', 'next_attraction']).size().reset_index(name='popularity')

# Normalize popularity scores
transition_counts['popularity'] = transition_counts['popularity'] / transition_counts['popularity'].sum()

# Get unique POIs from the data
pois = np.append(df['poi'].astype(int).unique(), [301,302,303])

# Create all possible pairs of POIs
all_pairs = pd.DataFrame(list(product(pois, pois)), columns=['poi', 'next_attraction'])

# Merge to include all pairs, filling missing pairs with popularity = 0
transition_counts_full = all_pairs.merge(transition_counts, on=['poi', 'next_attraction'], how='left').fillna({'popularity': 0.1})

transition_counts_full.to_csv("data/transition_popularity.csv", index=False)


transition_counts_full.drop_duplicates(keep=False)
print(len(transition_counts_full))




