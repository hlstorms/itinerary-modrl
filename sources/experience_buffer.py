import pandas as pd
from datetime import *
from collections import deque
from tqdm import tqdm
import pickle

from load_data import *
from utils import *

#############################################
############ TRAINING EXPBUFFER #############
#############################################

def exp_buffer(env, str_exp):

    df_poi_h_train = df_poi_train.copy()
    grouped_df_h_train = df_poi_h_train.groupby(['id_veronacard', 'data_visita'])

    # One experience buffer for each context
    poi_exp_b = df_poi_it['id'].values.astype(str).tolist() # all poi ids
    time_input_exp_b =  [str(x) for x in range(0, 6)] # amount of hours to spend / 2
    rain_exp_b = ['rain', 'no_rain'] # rain or no rain
    temp_exp_b = [str(x) for x in range(0, 5)] #temperature (split in 5 sections)
    hol_exp_b = ['holiday', 'no_holiday'] # holiday or no holiday

    df_key_exp_buffer_h = pd.DataFrame({'poi': poi_exp_b})
    df_key_exp_buffer_h = df_key_exp_buffer_h.merge( pd.DataFrame({'time':time_input_exp_b}), how='cross')
    df_key_exp_buffer_h = df_key_exp_buffer_h.merge( pd.DataFrame({'temp':temp_exp_b}), how='cross')
    df_key_exp_buffer_h = df_key_exp_buffer_h.merge( pd.DataFrame({'rain':rain_exp_b}), how='cross')
    df_key_exp_buffer_h = df_key_exp_buffer_h.merge( pd.DataFrame({'hol':hol_exp_b}), how='cross')

    exp_buffer_h = {}
    for i in range(len(df_key_exp_buffer_h)):
        exp_buffer_h['+'.join(df_key_exp_buffer_h.loc[i].tolist())] = deque(maxlen=700)

    for (group_id, group_date), group_data in tqdm(grouped_df_h_train):
        if len(list(group_data['poi'].values)) != len(set(group_data['poi'].values)): #check if there are duplicates
            continue
        elif len(list(group_data['poi'].values)) == 1: # Check if itinerary only consists of one PoI
            continue
        
        poi_start_h = group_data['poi'].iloc[0]
        date_input_h = datetime.strptime(group_date, '%Y-%m-%d')
        
        # reset environment for each new user
        state = env.reset(poi_start_h, timedelta(hours=14), date_input_h, group_id)
        first_visit_time = None

        for index, row in group_data.iterrows():
            if first_visit_time is None:
                first_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
                last_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
                last_poi = row['poi']
            else:
                last_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
                last_poi = row['poi']
    
        # Define length of itinerary (last_visit_time + visit duration of last poi - first_visit_time) in hours
        time_input_h = (last_visit_time + timedelta(minutes = int(env.poi_time_visit[last_poi]))).hour - first_visit_time.hour
        if time_input_h // 2 > 5: 
            continue
        temp_h, rain_h = get_weather(date_input_h, df_weather)
        holiday = get_holiday(date_input_h)
        key_exp_buff = '+'.join([str(int(poi_start_h)), str(int(time_input_h/2)), str(temp_h), str(rain_h), str(holiday)])
        
        if len(exp_buffer_h[key_exp_buff]) >= 200:
            continue
        for index, row in group_data.iterrows():
            if (row['poi'] not in env.explored):
                act_space = env.action_space.copy()
                new_state, reward, done, _ = env.step(row['poi'])
                act_space_2 = env.action_space.copy()
                a = map_from_poi_to_action[row['poi']]
                exp_buffer_h[key_exp_buff].append([state, a, new_state, reward, done, act_space, act_space_2]) 
                state = new_state

        env.timeleft = timedelta(minutes=0)  # reset time left for each real user, they have no more time for visits

        # Check if all contexts have at least 100
        if all(len(v) >= 100 for v in exp_buffer_h.values()):  
            print('\nExperience buffer filled for all context\n')
            break

    # Save the replay buffer to a file
    with open('exp_buffer_' + str_exp + '.pkl', 'wb') as f:
        pickle.dump(exp_buffer_h, f)


    return exp_buffer_h