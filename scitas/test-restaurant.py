import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions_NLM import estimate_nested_logit, simulate_choice
from functions_geom_DIB import geom_DIB_on_alternatives

# Load data
data = pd.read_csv('./restaurants.dat', delimiter=',')

# Load data
data2 = pd.read_csv('./obs_choice.dat', delimiter=',')

beta_chinese_0 = 0.849
beta_ethiopian_0 = 0.489
beta_french_0 = 0.629
beta_indian_0 = 1.03
beta_japanese_0 = 1.28
beta_korean_0 = 0.753
beta_lebanese_0 = 0.853
beta_log_dist_0 = -0.602
beta_mexican_0 = 1.27
beta_price_0 = -0.4
beta_rating_0 = 0.743

import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Compute differences in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Apply Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance

# Initialize a dictionary to store utilities for all users
all_utilities = {}

# Iterate over each user
for user_idx in range(10000):
    # Get the latitude and longitude of the current user
    user_lat = data2['user_lat'][user_idx]
    user_lon = data2['user_lon'][user_idx]
    rest_lat = data['rest_lat']
    rest_lon = data['rest_lon']
    
    # Initialize a list to store utilities for the current user
    user_utilities = []
    
    # Iterate over each restaurant
    for i in range(100):
        # Compute the distance between the user and the restaurant
        distance = haversine(user_lat, user_lon, rest_lat[i], rest_lon[i])
        
        # Compute utility for the ith restaurant for the current user
        U_i = beta_chinese_0 * data['category_Chinese'][i] + \
              beta_ethiopian_0 * data['category_Ethiopian'][i] + \
              beta_french_0 * data['category_French'][i] + \
              beta_indian_0 * data['category_Indian'][i] + \
              beta_japanese_0 * data['category_Japanese'][i] + \
              beta_korean_0 * data['category_Korean'][i] + \
              beta_lebanese_0 * data['category_Lebanese'][i] + \
              beta_mexican_0 * data['category_Mexican'][i] + \
              beta_price_0 * data['price'][i] + \
              beta_rating_0 * data['rating'][i] + \
              beta_log_dist_0 * np.log(distance)
        
        # Append the utility to the list for the current user
        user_utilities.append(U_i)
    
    # Store the list of utilities for the current user in the dictionary
    all_utilities[user_idx] = user_utilities

# Now 'all_utilities' is a dictionary where keys are user indices and values are lists of utilities
# You can access the utilities associated with any specific user by using their index as the key, for example:
# utilities_for_user_3000 = all_utilities[3000]

# Initialize a dictionary to store utilities for all users
all_probabilities = {}

# Iterate over each user
for user_idx in range(10000):
    user_utilities = all_utilities[user_idx]

    # Compute the sum of exponentials of utilities
    sum_exp_utilities = sum([math.exp(U_i) for U_i in user_utilities])

    # Compute the probability of choosing each restaurant
    user_probabilities = [math.exp(U_i) / sum_exp_utilities for U_i in user_utilities]

    # Store the list of probabilities for the current user in the dictionary
    all_probabilities[user_idx] = user_probabilities

    # make a dataset with the probabilities of each restaurant for each user, restaurant in columns, users in rows
df_probabilities = pd.DataFrame(all_probabilities)

#transpose the dataframe
df_probabilities = df_probabilities.T

#change the name of each column for P_i with i the number of the restaurant
df_probabilities.columns = ['P_' + str(i) for i in range(100)]

#add a column with the choice of each user
df_probabilities['choice'] = data2['logit_0']

# Compute AIC, BIC and log-likelihood for the model
log_likelihood = 0
n_parameters = 11
n_observations = 10000

# Iterate over each user
for user_idx in range(10000):
    user_probabilities = all_probabilities[user_idx]
    user_choice = data2['logit_0'][user_idx]
    log_likelihood += math.log(user_probabilities[user_choice])

AIC = -2 * log_likelihood + 2 * n_parameters
BIC = -2 * log_likelihood + n_parameters * math.log(n_observations)

print('AIC:', AIC)
print('BIC:', BIC)
print('Log-likelihood:', log_likelihood)

df_input = data2[['user_lat', 'user_lon']]
# Calculate frequencies and probabilities
vector_counts = df_input.value_counts().reset_index(name='Frequency')
vector_counts['Probability'] = vector_counts['Frequency'] / len(df_input)

# Creating a tuple of attributes to facilitate mapping
vector_counts['tuple'] = vector_counts[['user_lat', 'user_lon']].apply(tuple, axis=1)
probability_map = vector_counts.set_index('tuple')['Probability'].to_dict()

# Assign the probability to each row based on its tuple of attributes
df_input['Probability'] = df_input.apply(lambda row: probability_map[tuple(row)], axis=1)

# Computation of p(x,y)
p_x = df_input['Probability'].values

p_y_given_x = np.array(list(all_probabilities.values()))
p_xy = p_x[:, np.newaxis] * p_y_given_x

# Normalize p_xy 
p_xy /= p_xy.sum()

# Define epsilon value
epsilon = 1e-20

# Add epsilon to elements equal to 0
p_xy[p_xy == 0] += epsilon

betas = np.linspace(0, 2500, 5)
# Initialize an empty list to store the number of clusters
num_clusters_list = []

# Iterate over each beta value
for beta in betas:
    # Run iterative_algorithm to obtain q_t_given_x
    q_t_given_x, _, _ = geom_DIB_on_alternatives(p_xy, max_iter=2000, beta=beta, threshold=1e-20)
    
    # Calculate the number of clusters
    column_sum = np.sum(q_t_given_x, axis=0)
    num_clusters = np.count_nonzero(column_sum)
    
    # Append the number of clusters to the list
    num_clusters_list.append(num_clusters)

# Plot the number of clusters against beta values
plt.plot(betas, num_clusters_list)
plt.xlabel('Beta')
plt.ylabel('Number of Clusters')
plt.title('Number of Clusters vs. Beta')
plt.grid(True)
plt.savefig('plot.png')