import pandas as pd
from datetime import *
import ast

from PoiEnv import poi_env
from utils import *
from load_data import *

def context_distance_popularity_baseline(date_input, time_input, poi_start, group_id, context=False):
    # choose the nearest and most popular POI to vist until time is over
    done = False
    poi_len = 0

    # Initialization of the environment -> reset for each date
    env = poi_env()
    env.reset(poi_start, timedelta( hours = time_input ), date_input, group_id)
    visited_poi = [poi_start]
    reward = 0 
    matches = 0 

    if context:
        temp, rain = get_weather(date_input, df_weather)
        poi_popularity = df_poi_popularity_context[(df_poi_popularity_context['temp'] == temp) & 
                                                   (df_poi_popularity_context['rain'] == rain)].copy()
    else:
        poi_popularity = df_poi_popularity.copy()
    
    while not done:   # check if all possible actions are done
        last_poi_visited = env.state[0]
        distance_poi = df_poi_time_travel[(df_poi_time_travel['poi_start'] == last_poi_visited) & 
                                            (df_poi_time_travel['poi_dest'] != last_poi_visited) & df_poi_time_travel['poi_dest'].isin(env.action_space)].copy()
        distance_values = list(distance_poi['time_travel'].drop_duplicates().sort_values())

        distance_poi['points'] = distance_poi['time_travel'].apply(lambda x: distance_values.index(x)+1)
        distance_poi.rename(columns={'poi_dest': 'poi'}, inplace=True)
        choice_poi = pd.merge(distance_poi, poi_popularity, on='poi', how='left')
        choice_poi['tot'] = choice_poi['points'] + choice_poi['position']
        choice_poi = choice_poi.sort_values(by=['tot'], ascending=True)

        a = choice_poi['poi'].values[0]
        
        _ , r, done, match = env.step(a)
        reward += r
        poi_len += 1
        matches += match
        visited_poi.append(a)

    total_time_visit, total_time_distance, total_time_crowd, time_left=env.time_stats()
    
    popular_poi_visited = len(set(visited_poi) & set(popular_poi))
        
    return reward, total_time_visit, total_time_distance, total_time_crowd, time_left, poi_len, popular_poi_visited, visited_poi, matches

def preferences_baseline(date_input, time_input, poi_start, group_id):
    # choose the nearest and most popular POI to vist until time is over
    done = False
    poi_len = 0

    # Initialization of the environment -> reset for each date
    env = poi_env()
    env.reset(poi_start, timedelta( hours = time_input ), date_input, group_id)
    visited_poi = [poi_start]
    reward = 0 
    matches = 0 
    
    while not done:   # check if all possible actions are done
        preferences = ast.literal_eval(df_preferences[df_preferences['id_veronacard'] == group_id]['preferences'].values[0])
        all_preferences = []
        for poi_dest in env.action_space:    
            tags = df_poi_it[df_poi_it['id'] == poi_dest]['tags'].iloc[0].split(", ")
            tags_binary = [1 if category in tags else 0 for category in ['Architecture','Arts','History', 'Nature', 'Religious Sites']]
            all_preferences.append(sum([(preferences[x])*tags_binary[x] for x in range(0,5)]))


        a_index = np.array(all_preferences).argmax()
        a = list(env.action_space)[a_index]
        
        _ , r, done, match = env.step(a)
        reward += r
        poi_len += 1
        matches += match
        visited_poi.append(a)

    total_time_visit, total_time_distance, total_time_crowd, time_left=env.time_stats()
    
    popular_poi_visited = len(set(visited_poi) & set(popular_poi))
        
    return reward, total_time_visit, total_time_distance, total_time_crowd, time_left, poi_len, popular_poi_visited, visited_poi, matches