import pandas as pd
from utils import neural_poi_map, get_holiday, is_valid_date


#############################################
################ LOAD # DATA ################
#############################################

# Latitude, Longitude, Time visit
df_poi_it = pd.read_csv('data/poi_it_complete2.csv', usecols=['id', 'latitude', 'longitude', 'Time_Visit', 'max_crowd', 'tags'], delimiter=';' , encoding='latin-1')

# POI occupation grouped by holiday and time slots
df_crowding = pd.read_csv('data/crowding.csv', usecols=['poi', 'holiday', 'hour', 'val_stim'])

# Time travel in minutes of each pair of poi
df_poi_time_travel = pd.read_csv('data/poi_time_travel.csv', usecols=['poi_start', 'poi_dest', 'time_travel'])

# Popularity of the transition between each pair of poi
df_popularity = pd.read_csv('data/transition_popularity.csv', usecols=['poi', 'next_attraction', 'popularity'])

# User attraction preferences
df_preferences = pd.read_csv('data/user_preferences.csv', usecols=['id_veronacard', 'preferences'])

# weather condition during 2022 [temperature,rain]
df_weather = pd.read_csv('data/weather_data.csv', usecols=['date','temp','rain'])

# data containing user visits
df_poi_train = pd.read_csv('data/train_data.csv', usecols=["id_veronacard","profilo","data_visita","ora_visita","sito_nome","poi"]).sort_values(
    ['id_veronacard', 'data_visita', 'ora_visita'])[:100000]
df_poi_train = df_poi_train[df_poi_train["data_visita"].apply(is_valid_date)].copy()

df_poi_test = pd.read_csv('data/test_data_sampled.csv', usecols=["id_veronacard","profilo","data_visita","ora_visita","sito_nome","poi"]).sort_values(
    ['id_veronacard', 'data_visita', 'ora_visita'])

# POI popularity 
df_poi_popularity = pd.read_csv('data/poi_popularity_train.csv', usecols=['poi', 'popularity','position'])
df_poi_popularity_context = pd.read_csv('data/poi_popularity_ctx_train.csv', usecols=['temp','rain','poi','popularity','position'])

# df_poi_popularity_test = pd.read_csv('data/poi_popularity_2023.csv', usecols=['poi', 'popularity','position'])
# df_poi_popularity_context_test = pd.read_csv('data/poi_popularity_ctx_2023.csv', usecols=['temp','rain','poi','popularity','position'])

popular_poi = df_poi_popularity.sort_values(by=['popularity'], ascending=False)['poi'].values[:3]


# map POI->action
map_from_poi_to_action, map_from_action_to_poi = neural_poi_map()