import numpy as np
import random
from datetime import *
from tensorflow import keras
from keras import layers

from utils import *
from load_data import map_from_poi_to_action, map_from_action_to_poi

#############################################
################ CLASS # DQN ################
#############################################

def initialization_dqn(layer_size, input_layer, output_layer):
    model = keras.Sequential()
    model.add(layers.Dense(layer_size, input_dim=input_layer, activation="relu"))  # input layer + 1 hidden layer
    model.add(layers.Dense(layer_size, activation="relu"))  # 2
    model.add(layers.Dense(layer_size, activation="relu"))  # 3
    model.add(layers.Dense(layer_size, activation="relu"))  # 4
    model.add(layers.Dense(layer_size, activation="relu"))  # 5
    model.add(layers.Dense(output_layer, activation="linear"))  # output layer
    model.compile(loss="mean_squared_error", optimizer='adam')
    return model


def train_model_dqn(model, memory, batch_size, poi_set_len, gamma=0.96):
    size = min(batch_size, len(memory))
    mb = random.sample(memory, size) #Sample random mini match from memory / experience buffer

    for [s, a, s_1, r, done, s_act_space, s1_act_space] in mb:
        state = s.reshape(1, poi_set_len + 2)
        target = model.predict(state, verbose=0)
        target = target[0]
        for i in range(poi_set_len):
            if map_from_action_to_poi[i] not in s_act_space:
                target[i] = 0
        if done == True:
            target[a] = r
        else:
            state_1 = s_1.reshape(1, poi_set_len + 2)
            predict_state_1 = model.predict(state_1, verbose=0)[0]
            for i in range(poi_set_len):
                if map_from_action_to_poi[i] not in s1_act_space:
                    predict_state_1[i] = 0
            max_q = max(predict_state_1)
            target[a] = r + max_q * gamma
        model.fit(state, np.array([target]), epochs=15, verbose=0)

    return model


def DQN(environment, neural_network, trials, batch_size, time_input, poi_start, date_input, group_id,
        experience_buffer, epsilon_decay=0.97):
    
    if len(experience_buffer) > 100:
        experience_buffer = random.sample(experience_buffer, 100)
    epsilon = 1
    epsilon_min = 0.01
    score_queue = []
    score_max = 0
    best_journey = []
    best_trial = -1
    match_max = -1

    
    for trial in range(trials):
        s = environment.reset(poi_start, timedelta(hours=time_input), date_input, group_id)
        s_act_space = environment.action_space.copy()
        done = False
        score = 0
        matches = 0
        visited_poi = []
        episode_rewards = []
        
        while done == False:  # check if all actions are done
            if np.random.random() < epsilon: # ---> becomes less likely to happen as #trials increases
                a = random.choices(list(environment.action_space), k=1)[0]
                
            else:
                state = s.reshape(1, len(environment.poi_set) + 2)
                prediction = neural_network.predict(state, verbose=0)
                for i in range(len(environment.poi_set)):
                    if map_from_action_to_poi[i] not in environment.action_space:
                        prediction[0][i] = -1000000
                a_index = prediction.argmax()
                a = map_from_action_to_poi[a_index]
            
            state = s.reshape(1, len(environment.poi_set) + 2)
            prediction = neural_network.predict(state, verbose=0)
            for i in range(len(environment.poi_set)):
                if map_from_action_to_poi[i] not in environment.action_space:
                    prediction[0][i] = -1000000
            a_index = prediction.argmax()
            a = map_from_action_to_poi[a_index]
            visited_poi.append(a)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            s_1, r, done, match = environment.step(a)

            a = map_from_poi_to_action[a]
            s1_act_space = environment.action_space.copy()
            experience_buffer.append([s, a, s_1, r, done, s_act_space, s1_act_space])
            train_model_dqn(neural_network, experience_buffer, batch_size, len(environment.poi_set))
            s = s_1
            score += r
            matches += match
            s_act_space = s1_act_space.copy()
            episode_rewards.append(r)

        if score > score_max:
            score_max = score
            best_journey = visited_poi.copy()
            best_trial = trial
            match_max = matches
        score_queue.append(episode_rewards)

    return best_trial, best_journey, score_queue, match_max