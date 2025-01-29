import numpy as np
import pandas as pd
from datetime import *
from tqdm import tqdm
import matplotlib.pyplot as plt

from PoiEnv import poi_env #DQN and historical
from PoiEnv2 import poi_env2 #MODRL
from experience_buffer import *
from baselines import *
from DQN import *
from MP_L import *
from MP_NL import *
from SP_L import *
from SP_NL import *

#############################################
############ TRAINING EXPBUFFERS ############
#############################################

print("\n################## EXP BUFFER ##################\n")
exp_buffer_h = exp_buffer(env = poi_env(), str_exp="h")
exp_buffer_t = exp_buffer(env = poi_env2(), str_exp="t")

with open('exp_buffer_h.pkl', 'rb') as f:
    exp_buffer_h = pickle.load(f)
with open('exp_buffer_t.pkl', 'rb') as f:
    exp_buffer_mp_l = pickle.load(f)
with open('exp_buffer_t.pkl', 'rb') as f:
    exp_buffer_mp_nl = pickle.load(f)
with open('exp_buffer_h.pkl', 'rb') as f:
    exp_buffer_sp_l = pickle.load(f)
with open('exp_buffer_h.pkl', 'rb') as f:
    exp_buffer_sp_nl = pickle.load(f)
print("Experience buffer loaded from memory.")

#############################################
############ INITIALIZATIONS ################
#############################################

itinerary_h = pd.DataFrame(columns=['id_veronacard', 'date', 'itinerary', 'reward', 'time_visit', 'time_distance',
                                    'time_crowd', 'time_left', 'time_input', 'popular_poi_visited', 'poi_len', 'preference_matches'])
itinerary_cdp = pd.DataFrame(columns=['id_veronacard', 'date', 'itinerary', 'reward', 'time_visit', 'time_distance',
                                    'time_crowd', 'time_left', 'time_input', 'popular_poi_visited', 'poi_len', 'preference_matches'])
itinerary_up = pd.DataFrame(columns=['id_veronacard', 'date','itinerary', 'reward', 'time_visit', 'time_distance',
                                    'time_crowd', 'time_left', 'time_input', 'popular_poi_visited', 'poi_len', 'preference_matches'])
itinerary_dqn = pd.DataFrame(columns=['id_veronacard', 'date','trial', 'itinerary', 'time_visit', 'time_distance',
                                      'time_crowd', 'time_left', 'time_input', 'popular_poi_visited', 'poi_len', 'scores', 'preference_matches','time_process'])
itinerary_mp_l = pd.DataFrame(columns=['id_veronacard', 'date','trial', 'itinerary', 'time_visit', 'time_distance',
                                      'time_crowd', 'time_left', 'time_input', 'popular_poi_visited', 'poi_len', 'scores', 'scal_scores', 'hypervolume', 'preference_matches','time_process'])
itinerary_sp_l = pd.DataFrame(columns=['id_veronacard', 'date','trial', 'itinerary', 'time_visit', 'time_distance',
                                      'time_crowd', 'time_left', 'time_input', 'popular_poi_visited', 'poi_len', 'scores', 'scal_scores', 'hypervolume', 'preference_matches','time_process'])
itinerary_mp_nl = pd.DataFrame(columns=['id_veronacard','date', 'trial', 'itinerary', 'time_visit', 'time_distance',
                                      'time_crowd', 'time_left', 'time_input', 'popular_poi_visited', 'poi_len', 'scores', 'scal_scores', 'hypervolume', 'preference_matches','time_process'])
itinerary_sp_nl = pd.DataFrame(columns=['id_veronacard', 'date','trial', 'itinerary', 'time_visit', 'time_distance',
                                      'time_crowd', 'time_left', 'time_input', 'popular_poi_visited', 'poi_len', 'scores', 'scal_scores', 'hypervolume', 'preference_matches','time_process'])

global_reward_h = global_reward_cdp = global_reward_up = global_reward_dqn = global_best_reward_dqn = global_reward_mp_l = global_best_reward_mp_l = global_reward_sp_l = global_best_reward_sp_l = global_reward_mp_nl = global_best_reward_mp_nl = global_reward_sp_nl = global_best_reward_sp_nl =  0
global_total_time_visit_h = global_total_time_visit_cdp = global_total_time_visit_up = global_total_time_visit_dqn = global_total_time_visit_mp_l = global_total_time_visit_sp_l = global_total_time_visit_mp_nl = global_total_time_visit_sp_nl = 0
global_total_time_distance_h = global_total_time_distance_cdp = global_total_time_distance_up = global_total_time_distance_dqn = global_total_time_distance_mp_l = global_total_time_distance_sp_l =  global_total_time_distance_mp_nl = global_total_time_distance_sp_nl =  0
global_total_time_crowd_h = global_total_time_crowd_cdp = global_total_time_crowd_up = global_total_time_crowd_dqn = global_total_time_crowd_mp_l = global_total_time_crowd_sp_l = global_total_time_crowd_mp_nl = global_total_time_crowd_sp_nl = 0
global_time_left_h = global_time_left_cdp = global_time_left_up = global_time_left_dqn = global_time_left_mp_l = global_time_left_sp_l =  global_time_left_mp_nl = global_time_left_sp_nl= 0
global_poi_len_h = global_poi_len_cdp = global_poi_len_up = global_poi_len_dqn = global_poi_len_mp_l = global_poi_len_sp_l = global_poi_len_mp_nl = global_poi_len_sp_nl = 0
global_matches_h = global_matches_cdp = global_matches_up = global_matches_dqn = global_matches_mp_l = global_matches_sp_l= global_matches_mp_nl = global_matches_sp_nl = 0
global_popular_poi_visited_h = global_popular_poi_visited_cdp = global_popular_poi_visited_up = global_popular_poi_visited_dqn = global_popular_poi_visited_mp_l = global_popular_poi_visited_sp_l = global_popular_poi_visited_mp_nl = global_popular_poi_visited_sp_nl= 0

global_time_h = global_time_process_dqn = global_time_process_mp_l = global_time_process_sp_l = global_time_process_mp_nl = global_time_process_sp_nl = 0
i_h = i_dqn = i_mp_l = i_sp_l = i_mp_nl = i_sp_nl = 0

#############################################
################# BASELINES #################
#############################################

print("\n################## BASELINES ##################\n")

# ### Load historical data
df_poi_h_test = df_poi_test.copy()
grouped_df_h_test = df_poi_h_test.groupby(['id_veronacard', 'data_visita'])
# Retrieve sample of users to get a mix of contextual factors 
random.seed(42)
sampled_groups = random.sample(list(grouped_df_h_test), k=100)
len_vc = len(grouped_df_h_test)
print(f"Number of users: {len_vc}")
# Initialization of the environment
env_h = poi_env()

for (group_id, group_date), group_data in tqdm(sampled_groups):
    group_id = str(group_id)

    if len(list(group_data['poi'].values)) != len(set(group_data['poi'].values)): #check if there are duplicates
        continue
    elif len(list(group_data['poi'].values)) == 1: # Check if itinerary only consists of one PoI
        continue 
    elif i_h == 20: 
        break
    poi_start = int(group_data['poi'].iloc[0])
    date_input = datetime.strptime(group_date, "%Y-%m-%d")
    
    # reset environment for each new user
    state = env_h.reset(poi_start, timedelta(hours=14), date_input, group_id) #TODO: why 14 hours?
    reward_tot = 0
    matches_h = 0
    relative_start_time = 0
    first_visit_time = None
    poi_len = 0

    for index, row in group_data.iterrows():
        poi_len += 1
        if (row['poi'] not in env_h.explored):
            if first_visit_time is None:
                first_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
                last_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
                last_poi = row['poi']

                env_h.state[1] = relative_start_time
            else:
                last_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
                last_poi = row['poi']

            act_space = env_h.action_space.copy()
            new_state, reward, done, match = env_h.step(row['poi'])

            act_space_2 = env_h.action_space.copy()
            a = map_from_poi_to_action[row['poi']]
            state = new_state
            reward_tot += reward
            matches_h += match
    env_h.timeleft = timedelta(minutes=0)  # reset time left for each real user, they have no more time for visits

    # Time_input => time between leaving last attraction and entering first attraction
    time_input = (last_visit_time + timedelta(minutes=env_h.poi_time_visit[last_poi])).hour - first_visit_time.hour
    if time_input not in [4,6,8]:
        continue
    total_time_visit_h, total_time_distance_h, total_time_crowd_h, time_left_h = env_h.time_stats()

    itinerary_h.loc[i_h] = [group_id, group_date, list(group_data['poi'].values), reward_tot, total_time_visit_h, total_time_distance_h, total_time_crowd_h, time_left_h, time_input, len(set(env_h.explored) & set(popular_poi)), poi_len, matches_h]
       
    global_total_time_visit_h += total_time_visit_h
    global_total_time_distance_h += total_time_distance_h
    global_total_time_crowd_h += total_time_crowd_h
    global_time_left_h += time_left_h
    global_time_h += time_input*60

    visited_poi = env_h.explored.copy()
    global_popular_poi_visited_h += len(set(visited_poi) & set(popular_poi))

    global_reward_h += reward_tot
    i_h += 1
    global_poi_len_h += poi_len
    global_matches_h += matches_h 

    ################### CDP ###################

    reward_cdp, total_time_visit_cdp, total_time_distance_cdp, total_time_crowd_cdp, time_left_cdp, poi_len_cdp, popular_poi_visited_cdp, poi_visited_cdp, matches_cdp = context_distance_popularity_baseline(date_input, time_input, poi_start, group_id, True)

    itinerary_cdp.loc[i_h] = [group_id, group_date, poi_visited_cdp, reward_cdp, total_time_visit_cdp, total_time_distance_cdp, total_time_crowd_cdp, time_left_cdp, time_input, popular_poi_visited_cdp, poi_len_cdp, matches_cdp]
 
    global_reward_cdp += reward_cdp
    global_total_time_visit_cdp += total_time_visit_cdp
    global_total_time_distance_cdp += total_time_distance_cdp
    global_total_time_crowd_cdp += total_time_crowd_cdp
    global_time_left_cdp += time_left_cdp
    global_poi_len_cdp += poi_len_cdp
    global_popular_poi_visited_cdp += popular_poi_visited_cdp
    global_matches_cdp += matches_cdp

    ################### UP ###################

    reward_up, total_time_visit_up, total_time_distance_up, total_time_crowd_up, time_left_up, poi_len_up, popular_poi_visited_up, poi_visited_up, matches_up = preferences_baseline(date_input, time_input, poi_start, group_id)
    itinerary_up.loc[i_h] = [group_id, group_date, poi_visited_up, reward_up, total_time_visit_up, total_time_distance_up, total_time_crowd_up, time_left_up, time_input, popular_poi_visited_up, poi_len_up, matches_up]
 
    global_reward_up += reward_up
    global_total_time_visit_up += total_time_visit_up
    global_total_time_distance_up += total_time_distance_up
    global_total_time_crowd_up += total_time_crowd_up
    global_time_left_up += time_left_up
    global_poi_len_up += poi_len_up
    global_popular_poi_visited_up += popular_poi_visited_up
    global_matches_up += matches_up


    ################### DQN ###################

    env_dqn = poi_env()
    start_state_dqn = env_dqn.reset(poi_start, timedelta( hours = time_input ) , date_input, group_id)
    neural_network = initialization_dqn(layer_size=15, input_layer=20, output_layer=18)

    temp, rain = get_weather(date_input, df_weather)
    holiday = get_holiday(date_input)
    key_exp_buff = '+'.join([str(int(poi_start)), str(int(time_input/2)), str(temp), str(rain), str(holiday)])
    
    experience_buffer_dqn = exp_buffer_h[key_exp_buff]
    if experience_buffer_dqn == 'DONE':
        continue
    else:
        i_dqn += 1
    
    start_dqn = datetime.now()
   
    best_trial_dqn, best_journey_dqn, score_dqn, matches_dqn = DQN(env_dqn, neural_network, 30, 32, time_input, poi_start, date_input, group_id, experience_buffer_dqn)
    time_process_dqn = (datetime.now() - start_dqn).total_seconds()
    poi_len_dqn = len(best_journey_dqn)
    global_matches_dqn += matches_dqn

    exp_buffer_h[key_exp_buff] = 'DONE'

    total_time_visit_dqn, total_time_distance_dqn, total_time_crowd_dqn, time_left_dqn=env_dqn.time_stats()
    popular_poi_visited_dqn = len(set(best_journey_dqn) & set(popular_poi))
    global_popular_poi_visited_dqn += popular_poi_visited_dqn
    global_total_time_visit_dqn += total_time_visit_dqn
    global_total_time_distance_dqn += total_time_distance_dqn
    global_total_time_crowd_dqn += total_time_crowd_dqn
    global_time_left_dqn += time_left_dqn
    global_poi_len_dqn += poi_len_dqn
    global_time_process_dqn += time_process_dqn

    itinerary_dqn.loc[i_dqn] = [group_id, group_date, best_trial_dqn, list(best_journey_dqn), total_time_visit_dqn, total_time_distance_dqn, total_time_crowd_dqn, 
                             time_left_dqn, time_input, popular_poi_visited_dqn, len(best_journey_dqn), score_dqn, matches_dqn, time_process_dqn]


    ################### MP Linear ###################


    # Initialization of the environment
    env_mp_l = poi_env2()
    start_state_mp_l = env_mp_l.reset(poi_start, timedelta( hours = time_input ) , date_input, group_id)

    # Initialization of the neural network
    neural_network = initialization_mp_l(layer_size=15, input_layer=20, poi_set_len=18, num_objectives=2)

    # run MODQN algorithm (deep q learning)
    experience_buffer_mp_l = exp_buffer_mp_l[key_exp_buff]
    if experience_buffer_mp_l == 'DONE':
        continue
    else:
        i_mp_l += 1

    start_mp_l = datetime.now()
    best_trial_mp_l, best_journey_mp_l, score_mp_l, score_scal_mp_l, hypervolume_mp_l, matches_mp_l, = MP_L(env_mp_l, neural_network, 30, 32, time_input, poi_start, date_input, group_id, experience_buffer_mp_l)
    time_process_mp_l = (datetime.now() - start_mp_l).total_seconds()
    poi_len_mp_l = len(best_journey_mp_l)
    global_matches_mp_l += matches_mp_l

    exp_buffer_h[key_exp_buff] = 'DONE'

    total_time_visit_mp_l, total_time_distance_mp_l, total_time_crowd_mp_l, time_left_mp_l=env_mp_l.time_stats()
    popular_poi_visited_mp_l = len(set(best_journey_mp_l) & set(popular_poi))
    global_popular_poi_visited_mp_l += popular_poi_visited_mp_l
    global_total_time_visit_mp_l += total_time_visit_mp_l
    global_total_time_distance_mp_l += total_time_distance_mp_l
    global_total_time_crowd_mp_l += total_time_crowd_mp_l
    global_time_left_mp_l += time_left_mp_l
    global_poi_len_mp_l += poi_len_mp_l
    global_time_process_mp_l += time_process_mp_l

    itinerary_mp_l.loc[i_mp_l] = [group_id, group_date, best_trial_mp_l ,list(best_journey_mp_l), total_time_visit_mp_l, total_time_distance_mp_l, total_time_crowd_mp_l, 
                             time_left_mp_l, time_input, popular_poi_visited_mp_l, poi_len_mp_l, score_mp_l, score_scal_mp_l, hypervolume_mp_l, matches_mp_l, time_process_mp_l]

    ################### SP Linear ###################

    # Initialization of the environment
    env_sp_l = poi_env2()
    start_state_sp_l = env_sp_l.reset(poi_start, timedelta( hours = time_input ) , date_input, group_id)

    # Initialization of the neural network
    neural_network = initialization_sp_l(layer_size=15, input_layer=20, output_layer=18)

    # run MODQN algorithm (deep q learning)
    experience_buffer_sp_l = exp_buffer_sp_l[key_exp_buff]

    if experience_buffer_sp_l == 'DONE':
        continue
    else:
        i_sp_l += 1

    start_sp_l = datetime.now()
    best_trial_sp_l, best_journey_sp_l, score_sp_l, scal_score_sp_l, hypervolume_sp_l, matches_sp_l = SP_L(env_sp_l, neural_network, 5, 32, time_input, poi_start, date_input, group_id, experience_buffer_sp_l)
    time_process_sp_l = (datetime.now() - start_sp_l).total_seconds()
    poi_len_sp_l = len(best_journey_sp_l)
    global_matches_sp_l += matches_sp_l

    exp_buffer_sp_l[key_exp_buff] = 'DONE' 

    total_time_visit_sp_l, total_time_distance_sp_l, total_time_crowd_sp_l, time_left_sp_l = env_sp_l.time_stats()
    popular_poi_visited_sp_l = len(set(best_journey_sp_l) & set(popular_poi))
    global_popular_poi_visited_sp_l += popular_poi_visited_sp_l
    global_total_time_visit_sp_l += total_time_visit_sp_l
    global_total_time_distance_sp_l += total_time_distance_sp_l
    global_total_time_crowd_sp_l += total_time_crowd_sp_l
    global_time_left_sp_l += time_left_sp_l
    global_poi_len_sp_l += poi_len_sp_l
    global_time_process_sp_l += time_process_sp_l

    itinerary_sp_l.loc[i_sp_l] = [group_id, group_date, best_trial_sp_l, list(best_journey_sp_l), total_time_visit_sp_l, total_time_distance_sp_l, total_time_crowd_sp_l, 
                             time_left_sp_l, time_input, popular_poi_visited_sp_l, poi_len_sp_l, score_sp_l, scal_score_sp_l, hypervolume_sp_l, matches_sp_l, time_process_sp_l]

    ################### MP Non-Linear ###################


    # Initialization of the environment
    env_mp_nl = poi_env2()
    start_state_mp_nl = env_mp_nl.reset(poi_start, timedelta( hours = time_input ) , date_input, group_id)

    # Initialization of the neural network
    neural_network = initialization_mp_nl(layer_size=15, input_layer=20, poi_set_len=18, num_objectives=2)

    # run MODQN algorithm (deep q learning)
    experience_buffer_mp_nl = exp_buffer_mp_nl[key_exp_buff]
    if experience_buffer_mp_nl == 'DONE':
        continue
    else:
        i_mp_nl += 1

    start_mp_nl = datetime.now()
    best_trial_mp_nl, best_journey_mp_nl, score_mp_nl, score_scal_mp_nl, hypervolume_mp_nl, matches_mp_nl= MP_NL(env_mp_nl, neural_network, 30, 32, time_input, poi_start, date_input, group_id, experience_buffer_mp_nl)
    time_process_mp_nl = (datetime.now() - start_mp_nl).total_seconds()
    poi_len_mp_nl = len(best_journey_mp_nl)
    global_matches_mp_nl += matches_mp_nl

    exp_buffer_mp_nl[key_exp_buff] = 'DONE'

    total_time_visit_mp_nl, total_time_distance_mp_nl, total_time_crowd_mp_nl, time_left_mp_nl=env_mp_nl.time_stats()
    popular_poi_visited_mp_nl = len(set(best_journey_mp_nl) & set(popular_poi))
    global_popular_poi_visited_mp_nl += popular_poi_visited_mp_nl
    global_total_time_visit_mp_nl += total_time_visit_mp_nl
    global_total_time_distance_mp_nl += total_time_distance_mp_nl
    global_total_time_crowd_mp_nl += total_time_crowd_mp_nl
    global_time_left_mp_nl += time_left_mp_nl
    global_poi_len_mp_nl += poi_len_mp_nl
    global_time_process_mp_nl += time_process_mp_nl

    itinerary_mp_nl.loc[i_mp_nl] = [group_id, group_date, best_trial_mp_nl ,list(best_journey_mp_nl), total_time_visit_mp_nl, total_time_distance_mp_nl, total_time_crowd_mp_nl, 
                             time_left_mp_nl, time_input, popular_poi_visited_mp_nl, poi_len_mp_nl, score_mp_nl, score_scal_mp_nl, hypervolume_mp_nl, matches_mp_nl, time_process_mp_nl]

    ################### SP Non-Linear ###################

    # Initialization of the environment
    env_sp_nl = poi_env2()
    start_state_sp_nl = env_sp_nl.reset(poi_start, timedelta( hours = time_input ) , date_input, group_id)

    # Initialization of the neural network
    neural_network = initialization_sp_nl(layer_size=15, input_layer=20, output_layer=18)

    # run MODQN algorithm (deep q learning)
    experience_buffer_sp_nl = exp_buffer_sp_nl[key_exp_buff]

    if experience_buffer_sp_nl == 'DONE':
        continue
    else:
        i_sp_nl += 1

    start_sp_nl = datetime.now()
    best_trial_sp_nl, best_journey_sp_nl, score_sp_nl, scal_score_sp_nl, hypervolume_sp_nl, matches_sp_nl,  = SP_NL(env_sp_nl, neural_network, 30, 32, time_input, poi_start, date_input, group_id, experience_buffer_sp_nl)
    time_process_sp_nl = (datetime.now() - start_sp_nl).total_seconds()
    poi_len_sp_nl = len(best_journey_sp_nl)
    global_matches_sp_nl += matches_sp_nl

    exp_buffer_sp_nl[key_exp_buff] = 'DONE' 

    total_time_visit_sp_nl, total_time_distance_sp_nl, total_time_crowd_sp_nl, time_left_sp_nl = env_sp_nl.time_stats()
    popular_poi_visited_sp_nl = len(set(best_journey_sp_nl) & set(popular_poi))
    global_popular_poi_visited_sp_nl += popular_poi_visited_sp_nl
    global_total_time_visit_sp_nl += total_time_visit_sp_nl
    global_total_time_distance_sp_nl += total_time_distance_sp_nl
    global_total_time_crowd_sp_nl += total_time_crowd_sp_nl
    global_time_left_sp_nl += time_left_sp_nl
    global_poi_len_sp_nl += poi_len_sp_nl
    global_time_process_sp_nl += time_process_sp_nl

    itinerary_sp_nl.loc[i_sp_nl] = [group_id, group_date, best_trial_sp_nl, list(best_journey_sp_nl), total_time_visit_sp_nl, total_time_distance_sp_nl, total_time_crowd_sp_nl, 
                          time_left_sp_nl, time_input, popular_poi_visited_sp_nl, poi_len_sp_nl, score_sp_nl, scal_score_sp_nl, hypervolume_sp_nl, matches_sp_nl, time_process_sp_nl]

# Save itineraries 
with open('itinerary_h.pkl', 'wb') as f:
        pickle.dump(itinerary_h, f)
with open('itinerary_cdp.pkl', 'wb') as f:
        pickle.dump(itinerary_cdp, f)
with open('itinerary_up.pkl', 'wb') as f:
        pickle.dump(itinerary_up, f)                
with open('itinerary_dqn.pkl', 'wb') as f:
        pickle.dump(itinerary_dqn, f)
with open('itinerary_mp_l.pkl', 'wb') as f:
        pickle.dump(itinerary_mp_l, f)
with open('itinerary_sp_l.pkl', 'wb') as f:
        pickle.dump(itinerary_sp_l, f)
with open('itinerary_mp_nl.pkl', 'wb') as f:
        pickle.dump(itinerary_mp_nl, f) 
with open('itinerary_sp_nl.pkl', 'wb') as f:
        pickle.dump(itinerary_sp_nl, f)               
        
