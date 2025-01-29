from datetime import *
from math import ceil
import numpy as np
import ast
import utils
from load_data import *


class poi_env:
    def __init__(self):

        self.action_space = set(df_poi_it.id)  # PoIs available 
        self.state = None  # [poi,timestamp] [visited PoIs]
        self.timeleft = None  
        self.explored = set()  # PoIs visited
        self.map_from_poi_to_action, self.map_from_action_to_poi = utils.neural_poi_map()
    

        # dataframes
        self.df_poi_it = df_poi_it
        self.df_crowding = df_crowding
        self.df_poi_time_travel = df_poi_time_travel
        self.df_popularity = df_popularity
        self.df_preferences = df_preferences

        poi_position = {}   # latitude and longitude
        poi_time_visit = {} # time visit of each poi
        for row in df_poi_it.itertuples():
            poi_position[row[1]] = (row[3], row[4]) # (latitude, longitude)
            poi_time_visit[row[1]] = row[5] # Time_Visit
       

        self.poi_time_visit = poi_time_visit 
        self.poi_position = poi_position  
        self.poi_set = set(df_poi_it.id)  

        # STATS
        self.total_time_visit = 0
        self.total_time_crowd = 0
        self.total_time_distance = 0

    def reset(self, initial_poi, total_time, date, user_id):
        # reset environment
        self.action_space = self.poi_set.copy()  
        self.state = np.array([initial_poi, 0])  
        self.poi_date = date  # date of the tour
        self.preferences = ast.literal_eval(self.df_preferences[self.df_preferences['id_veronacard'] == user_id]['preferences'].values[0])

        self.state = np.pad(self.state, (0, len(self.poi_set)),
                            mode='constant')  # state  [0..1.....0...1] 1 if I poi 2+n-th is visited, 0 otherwise
        self.timeleft = total_time  # time still available
        self.explored = set() 

        self.total_time_visit = 0
        self.total_time_crowd = 0
        self.total_time_distance = 0 

        return self.state.copy()

    # Move from poi A to poi B
    def step(self, poi_dest):

        # compute walking time A -> B
        time_distance = self.calc_distance(poi_dest)
        # compute waiting time B
        time_crowd = self.crowding_wait(poi_dest)
        time_left_int = self.timeleft.total_seconds() / 60
        # compute popularity of transition
        popularity = self.transition_popularity(poi_dest)

        # calculate visit time, if it exceeds keep only the part of time that I can use
        if ((time_crowd + time_distance + self.poi_time_visit[poi_dest]) > time_left_int):
            if time_crowd + time_distance > time_left_int:
                time_visit = time_left_int
            else:
                time_visit = time_left_int - (time_crowd + time_distance)
        else:
            time_visit = self.poi_time_visit[poi_dest]

            # STATS
        self.total_time_distance += time_distance
        self.total_time_crowd += time_crowd
        self.total_time_visit += time_visit

        ##### NEW: USER PREFERENCES MATCH #####

        # If any of the user preferences are in the attraction tags, it is considered a match
        tags = self.df_poi_it[self.df_poi_it['id'] == poi_dest]['tags'].iloc[0].split(", ")
        tags_binary = [1 if category in tags else 0 for category in ['Architecture','Arts','History', 'Nature', 'Religious Sites']]
        temp, rain = utils.get_weather(self.poi_date, df_weather)
        poi_popularity = df_poi_popularity_context[(df_poi_popularity_context['temp'] == temp) & 
                                                   (df_poi_popularity_context['poi'] == poi_dest) & 
                                                   (df_poi_popularity_context['rain'] == rain)]['popularity'].values[0]
        preference_match = sum(np.multiply(self.preferences, tags_binary))
        
        ##### Reward ####
        reward = popularity/(poi_popularity+0.1) + (time_visit) / (time_crowd + time_distance + time_visit) * time_visit / 5

        #### State update ####
        self.action_space.remove(poi_dest)  # remove B from available PoIs
        self.explored.add(poi_dest)  # add B to visited PoIs
        self.state[0] = poi_dest  # update state of current PoI
        self.state[2 + self.map_from_poi_to_action[poi_dest]] = 1  # update state of visited PoIs

        ### Total time required
        time_total = time_distance + time_crowd + time_visit

        # Update time left
        self.timeleft = timedelta(minutes=(time_left_int - time_total))
        self.state[1] = int(self.state[1]) + int(time_total)

        # check if I visited all PoIs
        done_temp = False
        actual_act_space = self.action_space.copy()
        for poi in actual_act_space:
            self.poi_available(poi)
        if len(self.action_space) == 0:
            done_temp = True
            if self.timeleft.total_seconds() > 0:  # if time is not zero, reward is decreased
                reward = reward - self.timeleft.total_seconds() / 1200

        return self.state.copy(), reward, done_temp, preference_match

    def poi_available(self, poi_dest):

        time_travel = self.calc_distance(poi_dest)
        time_queue = self.crowding_wait(poi_dest)
        minimum_time = ceil((self.poi_time_visit[
            poi_dest]) / 3)  # less than 1/3 of visit_time, no sense to visit the poi
        time_t_q = time_travel + time_queue + minimum_time

        # if the remaining time does not allow to enter the destination POI, then I remove it from the visitable ones
        if (self.timeleft.total_seconds() / 60 < time_t_q):
            self.action_space.remove(poi_dest)

    def crowding_wait(self, poi_dest):
        # find crowd range looking at day and hour
        date_c = timedelta(minutes=int(self.state[1])) + self.poi_date 
        if date_c.hour < 12:
            crowd_range = date_c.replace(hour=8, minute=0, second=0)
        elif date_c.hour >= 12 and date_c.hour < 16:
            crowd_range = date_c.replace(hour=12, minute=0, second=0)
        else:
            crowd_range = date_c.replace(hour=16, minute=0, second=0)
        
        # Retrieve crowd estimation by poi, holiday, and hour
        estimated_crowd = \
        self.df_crowding.loc[(self.df_crowding['poi'] == poi_dest) & (self.df_crowding['holiday'] == get_holiday(self.poi_date)) & (self.df_crowding['hour'] == crowd_range.hour)][
            "val_stim"].values
        # if there is no data, I assume a crowd of no queue
        if len(estimated_crowd) == 0:
            return 15
        else:
            estimated_crowd = estimated_crowd[0]
            if np.isnan(estimated_crowd):
                return 15
            # compute waiting time (visit time/2 * estimated crowd) / maximum capacity of the poi TODO: find out why this formula?
            crowd_wait = ((self.poi_time_visit[poi_dest] / 2) * estimated_crowd) / \
                         self.df_poi_it.loc[(self.df_poi_it['id'] == poi_dest)]["max_crowd"].values[0]
            # return minutes
            return ceil(crowd_wait)

    def calc_distance(self, poi_dest):
        # compute walking time A -> B
        df_tmp = self.df_poi_time_travel.loc[(self.df_poi_time_travel['poi_start'] == self.state[0])]
        df_tmp = df_tmp.loc[(df_tmp['poi_dest'] == poi_dest)]
        return int(df_tmp.to_numpy()[0][2])
    
    def transition_popularity(self, poi_dest):
        # popularity of transitioning from A to B
        df_tmp = self.df_popularity.loc[(self.df_popularity['poi'] == int(self.state[0]))]
        df_tmp = df_tmp.loc[(df_tmp['next_attraction'] == int(poi_dest))]
        if not df_tmp.empty:
            return df_tmp['popularity'].iloc[0]
        else:
            return 0

    def time_stats(self):
        return self.total_time_visit, self.total_time_distance, self.total_time_crowd, self.timeleft.total_seconds() / 60

    def current_time(self):
        return timedelta(minutes=int(self.state[1])) + self.poi_date