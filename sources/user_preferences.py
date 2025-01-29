# import pandas as pd

# # Load data
# df_poi_it = pd.read_csv('data/poi_it_complete2.csv', usecols=['id', 'tags'], delimiter=';', encoding='latin-1')
# df_users_train = pd.read_csv("data/train_data.csv", usecols=["id_veronacard", "poi"])
# df_users_test = pd.read_csv("data/data_2023.csv", usecols=["id_veronacard", "poi"])

# # Concatenate user data
# df_users = pd.concat([df_users_train, df_users_test], axis=0)

# # Merge with POI data
# visits_with_categories = df_users.merge(df_poi_it[['id', 'tags']], left_on='poi', right_on="id")

# # Clean up tags column: remove brackets, split by commas
# visits_with_categories['tags'] = visits_with_categories['tags'].str.replace(r'[\[\]]', '', regex=True).str.split(',')

# # Explode to separate each tag
# visits_with_categories = visits_with_categories.explode('tags')

# # Normalize each tag by stripping whitespace
# visits_with_categories['tags'] = visits_with_categories['tags'].str.strip()

# # Count category visits per user
# category_counts = visits_with_categories.groupby(['id_veronacard', 'tags']).size().reset_index(name='count')

# # Identify the most frequently visited categories
# max_counts = category_counts.groupby('id_veronacard')['count'].transform(max)
# user_preferences = category_counts[category_counts['count'] == max_counts]

# # Aggregate top categories into a comma-separated string for each user (taking top 2)
# user_preferences = (
#     user_preferences.groupby('id_veronacard')['tags']
#     .apply(lambda x: ','.join(x[:2]))  # Join top 2 tags with commas
#     .reset_index()
# )

# # Save to CSV
# user_preferences.to_csv("data/user_preferences.csv", index=False)

import pandas as pd

# Load data
df_poi_it = pd.read_csv('data/poi_it_complete2.csv', usecols=['id', 'tags'], delimiter=';', encoding='latin-1')
df_users_train = pd.read_csv("data/train_data.csv", usecols=["id_veronacard", "poi"])
df_users_test = pd.read_csv("data/data_2023.csv", usecols=["id_veronacard", "poi"])
df_users = pd.concat([df_users_train, df_users_test], axis=0)

# Merge with POI data
visits_with_categories = df_users.merge(df_poi_it[['id', 'tags']], left_on='poi', right_on="id")

# Clean up tags column: remove brackets, split by commas
visits_with_categories['tags'] = visits_with_categories['tags'].str.split(',')

# Explode to separate each tag
visits_with_categories = visits_with_categories.explode('tags')

# Normalize each tag by stripping whitespace
visits_with_categories['tags'] = visits_with_categories['tags'].str.strip()

# Get unique categories
all_categories = visits_with_categories['tags'].unique()

# Count category visits per user
category_counts = visits_with_categories.groupby(['id_veronacard', 'tags']).size().reset_index(name='visit_count')

# Ensure all categories are present for every user
users = category_counts['id_veronacard'].unique()
all_categories_df = pd.DataFrame({'tags': all_categories})
user_categories = pd.MultiIndex.from_product([users, all_categories], names=['id_veronacard', 'tags']).to_frame(index=False)

# Merge to fill in missing categories with 0 visit count
category_counts = user_categories.merge(category_counts, on=['id_veronacard', 'tags'], how='left').fillna({'visit_count': 0})

# Rank categories by frequency for each user
category_counts['rank'] = category_counts.groupby('id_veronacard')['visit_count'].rank(ascending=False, method='max')

# Map ranks to integers from 1 (least favorite) to 5 (most favorite)
category_counts['rank'] = category_counts.groupby('id_veronacard')['rank'].transform(lambda x: 6 - x).astype(int)
category_counts['rank'] = category_counts['rank'].clip(lower=1, upper=5)

# Create a list of preferences for each user
user_preferences = (
    category_counts.sort_values(by=['id_veronacard', 'tags'])
    .groupby('id_veronacard')
    .apply(lambda df: df.sort_values(by='tags')['rank'].tolist())
    .reset_index(name='preferences')
)

# Save to CSV
user_preferences.to_csv("data/user_preferences.csv", index=False)




