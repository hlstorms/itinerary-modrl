import numpy as np
import pandas as pd
import random
from datetime import *
from tensorflow import keras
from keras import layers, Sequential

from utils import *
from load_data import map_from_poi_to_action, map_from_action_to_poi

#############################################
################ CLASS # MP Linear ##########
#############################################

def initialization_mp_l(layer_size, input_layer, poi_set_len, num_objectives):
    model = Sequential()
    model.add(layers.Input(shape=(input_layer,)))  # Input layer
    model.add(layers.Dense(layer_size, activation="relu"))  # Hidden layer 1
    model.add(layers.Dense(layer_size, activation="relu"))  # Hidden layer 2
    model.add(layers.Dense(layer_size, activation="relu"))  # Hidden layer 3
    model.add(layers.Dense(layer_size, activation="relu"))  # Hidden layer 4
    model.add(layers.Dense(poi_set_len * num_objectives, activation="linear"))  # Output layer
    model.add(layers.Reshape((poi_set_len, num_objectives)))  # Reshape to (poi_set_len, num_objectives)
    
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model



def train_model_mp_l(model, memory, batch_size, poi_set_len, objectives, gamma=0.96):
    size = min(batch_size, len(memory))
    minibatch = random.sample(memory, size)  # Random mini-batch from memory

    for s, a, s_1, r, done, s_act_space, s1_act_space in minibatch:
        state = s.reshape(1, poi_set_len + 2)
        targets = model.predict(state, verbose=0)[0]  # Shape: (poi_set_len, 2)

        # Mask out invalid actions
        for i in range(poi_set_len):
            if map_from_action_to_poi[i] not in s_act_space:
                targets[i] = [0] * len(objectives)  # Zero-out all objectives for invalid actions

        for obj_idx, obj in enumerate(objectives):
            if done:
                # Assign terminal reward for the current objective
                targets[a][obj_idx] = r[obj_idx]
            else:
                next_state = s_1.reshape(1, poi_set_len + 2)
                next_q_values = model.predict(next_state, verbose=0)[0]  # Shape: (poi_set_len, 2)

                # Mask out invalid actions in the next state
                for i in range(poi_set_len):
                    if map_from_action_to_poi[i] not in s1_act_space:
                        next_q_values[i] = [-np.inf] * len(objectives)  # Set invalid actions to -inf

                # Compute max Q-value for the current objective
                max_next_q = np.max([q[obj_idx] for q in next_q_values if q[obj_idx] != -np.inf])
                targets[a][obj_idx] = r[obj_idx] + gamma * max_next_q

        # Train the model with updated targets
        model.fit(state, targets.reshape(1, poi_set_len, len(objectives)), epochs=1, verbose=0)

    return model


def MP_L(environment, neural_network, trials, batch_size, time_input, poi_start, date_input, group_id,
        experience_buffer, epsilon_decay=0.95):

    # Limit the size of the experience buffer
    if len(experience_buffer) > 100:
        experience_buffer = random.sample(experience_buffer, 100)
    
    epsilon = 0.99
    epsilon_min = 0.1
    score_max = -1e5
    best_journey = []
    best_trial = -1
    score_queue = []
    hypervolume_queue = []
    reward_queue = []
    match_max = 0
    alpha = 0.6
    weights = np.array([alpha, 1-alpha])
    reference_point = [-2.0, -2.0]  # Example reference point

    for trial in range(trials):
        # Reset environment and initialize variables
        s = environment.reset(poi_start, timedelta(hours=time_input), date_input, group_id)
        s_act_space = environment.action_space.copy()
        done = False
        score = 0
        matches = 0
        visited_poi = []
        pareto_front = [] 
        episode_rewards = []
        
        while not done:  # Run until the episode ends
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # Choose a random action
                a = random.choice(list(environment.action_space))
            else:
                # Choose the action with the highest Q-value
                state = s.reshape(1, len(environment.poi_set) + 2)
                prediction = neural_network.predict(state, verbose=0)
                
                for i in range(len(environment.poi_set)):
                    if map_from_action_to_poi[i] not in environment.action_space:
                        prediction[0][i] = -np.inf  # Mask unavailable actions   
                
                prediction_comb = np.array([np.dot(np.array(pred), weights) for pred in prediction[0]])  
                
                a_index = prediction_comb.argmax()
                a = map_from_action_to_poi[a_index]
                
            visited_poi.append(a)

            # Update epsilon (exploration decay)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Execute action in the environment
            s_1, r, done, match = environment.step(a)

            # Only add non-dominated rewards to the Pareto front
            if not is_dominated(np.array(r), pareto_front):
                pareto_front = [p for p in pareto_front if not is_dominated(p, [np.array(r)])]
                pareto_front.append(np.array(r))

            episode_rewards.append(r)  # Log intermediate rewards for this episode
            r_scalarized = np.dot(np.array(r), weights)

            a = map_from_poi_to_action[a]
            s1_act_space = environment.action_space.copy()

            # Store the experience in the buffer
            experience_buffer.append([s, a, s_1, r, done, s_act_space, s1_act_space])

            # Train the neural network
            train_model_mp_l(neural_network, experience_buffer, batch_size, len(environment.poi_set), 
                        objectives=['sustainability', 'satisfaction'], gamma=0.96)

            # Update state, score, and matches
            s = s_1 
            s_act_space = s1_act_space.copy()
            score += r_scalarized
            matches += match

        hypervolume = compute_hypervolume(pareto_front, reference_point)

        if score > score_max:
            score_max = score
            best_journey = visited_poi.copy()
            best_trial = trial
            match_max = match

        # Log performance for this trial
        reward_queue.append(episode_rewards)
        score_queue.append(score)
        hypervolume_queue.append(hypervolume)

    return best_trial, best_journey, reward_queue, score_queue, hypervolume_queue, match_max