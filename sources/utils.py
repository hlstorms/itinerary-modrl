import holidays
from datetime import * 
import numpy as np
from pymoo.decomposition.tchebicheff import Tchebicheff

# map each poi to an action with a number from 0 to (#POI - 1)
def neural_poi_map():
    # POI ids
    all_poi = [42, 49, 52, 54, 58, 59, 61, 62, 63, 71, 75, 76, 201, 202, 300, 301, 302, 303]

    map_from_poi_to_action = {}  # Poi to Action
    i = 0
    for x in all_poi:
        map_from_poi_to_action[x] = i
        i += 1
    map_from_action_to_poi = {}
    for x in range(len(all_poi)):
        map_from_action_to_poi[x] = all_poi[x]
    return map_from_poi_to_action, map_from_action_to_poi

def print_date_type(date_input, df_weather, id_verona_card):
    # get all holidays from the years 2014 until 2023 (incl)
    giorni_festivi = []
    for year in range(2014,2024):
        giorni_festivi.extend(holidays.country_holidays("Italy", subdiv='VR', years=year))

    # print day informations (weather, temperature, holidays)
    row_weather = df_weather.loc[df_weather['date'] == date_input.strftime('%Y-%m-%d')]
    if not row_weather.empty:
        temperatura = row_weather['temp'].iloc[0]
        condizione = str(row_weather['rain'].iloc[0])
        if condizione == "nan": condizione = "sun" # if no rain then it is sunny 
    else:
        temperatura = None
        condizione = None

    if date_input in giorni_festivi or date_input.weekday() == 6 or date_input.weekday() == 5 or \
        (date_input.day() == 14 and date_input.month() == 2):
        print(
            f"ID Verona Card: {id_verona_card}  Date of the visit: {date_input.strftime('%d-%m-%Y')} (public holiday)\t  Temperature°°: {temperatura}\t Weather: {condizione} ")
    else:
        print(
            f"ID Verona Card: {id_verona_card}  Date of the visit: {date_input.strftime('%d-%m-%Y')} (weekday)\t Temperature°: {temperatura}\t Weather: {condizione} ")


# Print stats about the itinerary
def print_stats(total_time_visit, total_time_distance, total_time_crowd, time_left, time_input, popular, poi_len, arp, gini,us, prefix=""):
    print(f"{prefix}Total time used: {total_time_visit} min")
    print(f"{prefix}Time walked: {total_time_distance} min")
    print(f"{prefix}Time in queue: {total_time_crowd} min")
    print(f"{prefix}Time left: {time_left} min")
    print(f"{prefix}Popular POI visited: {popular}")
    print(f"{prefix}ARP: {arp} ")
    print(f"{prefix}Gini: {gini} ")
    print(f"{prefix}US: {us} ")

    percentage_visit = total_time_visit * 100 / (time_input * 60)
    percentage_distance = total_time_distance * 100 / (time_input * 60)
    percentage_crowd = total_time_crowd * 100 / (time_input * 60)
    percentage_final = time_left * 100 / (time_input * 60)
    print("{}Percentage of total time used (VT): {:3.2f}%".format(prefix,percentage_visit))
    print("{}Percentage of time walked     (MT): {:3.2f}%".format(prefix,percentage_distance))
    print("{}Percentage of time in queue   (QT): {:3.2f}%".format(prefix,percentage_crowd))
    print("{}Percentage of time left       (RT): {:3.2f}%".format(prefix,percentage_final))
    
    print(f"{prefix}POI LEN = {poi_len}")

def get_weather(date_input, df_weather):
    
    temp_df = df_weather[df_weather['date'] == date_input.strftime("%Y-%m-%d")]['temp']
    if temp_df.empty:
        temp = 2
    else:
        temp = temp_df.values[0] 

    rain_df = df_weather[df_weather['date'] == date_input.strftime("%Y-%m-%d")]['rain']
    if rain_df.empty:
        rain = "no_rain"
    else:
        rain = rain_df.values[0] 
    return temp, rain

def get_holiday(date_input):
    # get all holidays from the years 2014 until 2023 (incl)
    giorni_festivi = []
    for year in range(2014,2024):
        giorni_festivi.extend(holidays.country_holidays("Italy", subdiv='VR', years=year))
    # Holiday is either saturday, sunday, national holiday or any day in july and august
    if date_input in giorni_festivi or date_input.weekday() == 6 or date_input.weekday() == 5 or \
        (date_input.day== 14 and date_input.month == 2) or date_input.month == 6 or date_input.month == 7:
        return 'holiday'
    else:
        return 'no_holiday' 


def arp_measure(poi_list, itinerary_list):
    arp = 0
    for poi in poi_list:
        cont = 0
        for itinerary in itinerary_list:
            if poi in itinerary:
                cont += 1
        arp += cont / len(itinerary_list)
    return arp / len(poi_list)

def gini_measure(poi_list, itinerary_list):
    gini = 0
    for p in poi_list:
        for q in poi_list:
            if p==q:
                continue
            cont_p = 0
            cont_q = 0
            for it in itinerary_list:
                if p in it:
                    cont_p += 1
                if q in it:
                    cont_q += 1
            diff = abs(cont_p - cont_q)
            gini += diff
    return gini / (len(poi_list) * len(poi_list) * len(itinerary_list))

def us_measure(itinerary_list):

    avg_satisfaction = (itinerary_list['preference_matches'] / \
                        (itinerary_list['poi_len']*(itinerary_list['time_crowd'] / (itinerary_list['time_input']*60)))).mean()

    return avg_satisfaction

def tchebicheff(tau: float, reward_dim: int):
    """Tchebicheff scalarization function.

    This function requires a reference point. It is automatically adapted to the best value seen so far for each component of the reward.

    Args:
        tau: Parameter to be sure the reference point is always dominating (automatically adapted).
        reward_dim: Dimension of the reward vector

    Returns:
        Callable: Tchebicheff scalarization function
    """
    best_so_far = [float("-inf") for _ in range(reward_dim)]
    tch = Tchebicheff()

    def thunk(reward: np.ndarray, weights: np.ndarray):
        for i, r in enumerate(reward):
            if best_so_far[i] < r + tau:
                best_so_far[i] = r + tau
        return -tch.do(F=reward, weights=weights, utopian_point=np.array(best_so_far))[0][0]

    return thunk

def compute_hypervolume(pareto_front, reference_point):
    """
    Compute the hypervolume of a Pareto front with respect to a reference point.

    Parameters:
    - pareto_front: A list of Pareto-optimal points (list of objectives).
    - reference_point: A point that dominates all Pareto front points.

    Returns:
    - hypervolume: The computed hypervolume.
    """
    hypervolume = 0
    sorted_front = sorted(pareto_front, key=lambda x: x[0])  # Sort by the first objective

    for i, point in enumerate(sorted_front):
        width = reference_point[0] - point[0]
        height = reference_point[1] - point[1]
        hypervolume += width * height

    return hypervolume

def is_valid_date(date_str):
    try:
        # Ensure the value is a string before checking
        if not isinstance(date_str, str):
            return False
        # Try to parse the date with the correct format
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False
    
def tlo_scalarization(rewards, thresholds):
    """
    Perform TLO scalarization by comparing rewards to thresholds.

    Args:
        rewards (np.array): Array of rewards for the objectives.
        thresholds (np.array): Thresholds for each objective.

    Returns:
        float: Scalarized reward value based on TLO.
    """
    for i, (reward, threshold) in enumerate(zip(rewards, thresholds)):
        if reward < threshold:
            return reward - i * 1e-3  # Penalize lower-priority objectives slightly
    return rewards[-1]  # If all thresholds are met, return the last objective reward

def is_dominated(point, pareto_front):
    """
    Check if a point is dominated by any point in the Pareto front.

    Args:
        point (np.array): The point to check.
        pareto_front (list of np.array): Current Pareto front.

    Returns:
        bool: True if the point is dominated, False otherwise.
    """
    for p in pareto_front:
        if all(p <= point) and any(p < point):
            return True
    return False