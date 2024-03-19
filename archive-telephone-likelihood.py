import numpy as np

# Likelihood function for 1,2 vs 3,5
# beta will be beta = ["ASC_1", "ASC_3", "ASC_5", "BETA_COST", "lambda_1", "lambda_2"]

def log_likelihood_telephone_12vs35(beta, data):
    # Define utility functions
    data['U_1'] = beta[0] + beta[3] * data['logcost1'] 
    data['U_2'] = beta[3] * data['logcost2'] 
    data['U_3'] = beta[1] + beta[3] * data['logcost3']
    data['U_5'] = beta[2] + beta[3] * data['logcost5']
    
    # Calculate logsum for nests with > 1 alt
    data['logsum_1'] = np.log(data['avail1'] * np.exp(data['U_1'] / beta[4])
                                        + data['avail2'] * np.exp(data['U_2'] / beta[4])
                                        + (1 - data['avail1']) * (1 - data['avail2']))
    data['logsum_2'] = np.log(data['avail3'] * np.exp(data['U_3'] / beta[5])
                                    + data['avail5'] * np.exp(data['U_5'] / beta[5])
                                    + (1 - data['avail3']) * (1 - data['avail5']))

    # Nest probabilities
    data['P_nest_1'] = np.exp(beta[4] * data['logsum_1']) / \
                                 (np.exp(beta[4] * data['logsum_1']) 
                                  + np.exp(beta[5] * data['logsum_2']))
    data['P_nest_2'] = 1 - data['P_nest_1']
    
    # Within nest probabilities for nests with > 1 alt
    data['P_1_in_nest1'] = data['avail1'] * np.exp(data['U_1'] / beta[4]) / \
                                (data['avail1'] * np.exp(data['U_1'] / beta[4]) 
                                 + data['avail2'] * np.exp(data['U_2'] / beta[4]))
    data['P_2_in_nest1'] = 1 - data['P_1_in_nest1']
    
    data['P_3_in_nest2'] = data['avail3'] * np.exp(data['U_3'] / beta[5]) / \
                            (data['avail3'] * np.exp(data['U_3'] / beta[5])
                                + data['avail5'] * np.exp(data['U_5'] / beta[5]))
    data['P_5_in_nest2'] = 1 - data['P_3_in_nest2']
    
    # Full probabilities
    data['P_1'] = data['P_nest_1'] * data['P_1_in_nest1']
    data['P_2'] = data['P_nest_1'] * data['P_2_in_nest1']
    data['P_3'] = data['P_nest_2'] * data['P_3_in_nest2']
    data['P_5'] = data['P_nest_2'] * data['P_5_in_nest2']
    
    # Calculate probability for chosen alternative for each row
    data['P'] = (data['choice'] == 1) * data['P_1'] + \
                (data['choice'] == 2) * data['P_2'] + \
                (data['choice'] == 3) * data['P_3'] + \
                (data['choice'] == 5) * data['P_5']
    
    # Replace zero probabilities with small value to avoid LL = -inf
    epsilon = 1e-20
    data.loc[data['P'] == 0, 'P'] = epsilon
    
    # Calculate log-likelihood 
    LL = data['P'].apply(np.log).sum()
    
    return -LL  # We minimize negative log-likelihood

# Define log-likelihood function 1,2 vs 3,4,5
# beta will be beta = ["ASC_1", "ASC_3", "ASC_4", "ASC_5", "BETA_COST", "lambda_measured", "lambda_flat"]

def log_likelihood_telephone(beta, data):
    # Define utility functions
    data['U_1'] = beta[0] + beta[4] * data['logcost1'] 
    data['U_2'] = beta[4] * data['logcost2'] 
    data['U_3'] = beta[1] + beta[4] * data['logcost3']
    data['U_4'] = beta[2] + beta[4] * data['logcost4']
    data['U_5'] = beta[3] + beta[4] * data['logcost5']
    
    # Calculate logsum for nests with > 1 alt
    data['logsum_measured'] = np.log(data['avail1'] * np.exp(data['U_1'] / beta[5])
                                        + data['avail2'] * np.exp(data['U_2'] / beta[5])
                                        + (1 - data['avail1']) * (1 - data['avail2']))
    data['logsum_flat'] = np.log(data['avail3'] * np.exp(data['U_3'] / beta[6])
                                    + data['avail4'] * np.exp(data['U_4'] / beta[6])
                                    + data['avail5'] * np.exp(data['U_5'] / beta[6])
                                    + (1 - data['avail3']) * (1 - data['avail4']) * (1 - data['avail5']))
    
    # Nest probabilities
    data['P_nest_measured'] = np.exp(beta[5] * data['logsum_measured']) / \
                                 (np.exp(beta[5] * data['logsum_measured']) 
                                  + np.exp(beta[6] * data['logsum_flat']))
    data['P_nest_flat'] = 1 - data['P_nest_measured']
    
    # Within nest probabilities for nests with > 1 alt
    data['P_1_in_measured'] = data['avail1'] * np.exp(data['U_1'] / beta[5]) / \
                                (data['avail1'] * np.exp(data['U_1'] / beta[5]) 
                                 + data['avail2'] * np.exp(data['U_2'] / beta[5]))
    data['P_2_in_measured'] = 1 - data['P_1_in_measured']

    data['P_3_in_flat'] = data['avail3'] * np.exp(data['U_3'] / beta[6]) / \
                            (data['avail3'] * np.exp(data['U_3'] / beta[6])
                                + data['avail4'] * np.exp(data['U_4'] / beta[6])
                                + data['avail5'] * np.exp(data['U_5'] / beta[6]))
    data['P_4_in_flat'] = data['avail4'] * np.exp(data['U_4'] / beta[6]) / \
                            (data['avail3'] * np.exp(data['U_3'] / beta[6])
                                + data['avail4'] * np.exp(data['U_4'] / beta[6])
                                + data['avail5'] * np.exp(data['U_5'] / beta[6]))
    data['P_5_in_flat'] = 1 - data['P_3_in_flat'] - data['P_4_in_flat']
    
    # Full probabilities
    data['P_1'] = data['P_nest_measured'] * data['P_1_in_measured']
    data['P_2'] = data['P_nest_measured'] * data['P_2_in_measured']
    data['P_3'] = data['P_nest_flat'] * data['P_3_in_flat']
    data['P_4'] = data['P_nest_flat'] * data['P_4_in_flat']
    data['P_5'] = data['P_nest_flat'] * data['P_5_in_flat']
    
    # Calculate probability for chosen alternative for each row
    data['P'] = (data['choice'] == 1) * data['P_1'] + \
                (data['choice'] == 2) * data['P_2'] + \
                (data['choice'] == 3) * data['P_3'] + \
                (data['choice'] == 4) * data['P_4'] + \
                (data['choice'] == 5) * data['P_5']
    
    # Replace zero probabilities with small value to avoid LL = -inf
    epsilon = 1e-20
    data[data['P'] == 0] = epsilon
    
    # Calculate log-likelihood 
    LL = data['P'].apply(np.log).sum()
    
    return -LL  # We minimize negative log-likelihood


# Define log-likelihood function for 1,2,3 vs 4,5
# beta will be beta = ["ASC_1", "ASC_3", "ASC_4", "ASC_5", "BETA_COST", "lambda_1", "lambda_2"]

def log_likelihood_telephone_123vs45(beta, data):
    # Define utility functions
    data['U_1'] = beta[0] + beta[4] * data['logcost1'] 
    data['U_2'] = beta[4] * data['logcost2'] 
    data['U_3'] = beta[1] + beta[4] * data['logcost3']
    data['U_4'] = beta[2] + beta[4] * data['logcost4']
    data['U_5'] = beta[3] + beta[4] * data['logcost5']
    
    # Calculate logsum for nests with > 1 alt
    data['logsum_1'] = np.log(data['avail1'] * np.exp(data['U_1'] / beta[5])
                                        + data['avail2'] * np.exp(data['U_2'] / beta[5])
                                        + data['avail3'] * np.exp(data['U_3'] / beta[5])
                                        + (1 - data['avail1']) * (1 - data['avail2']) * (1 - data['avail3']))
    data['logsum_2'] = np.log(data['avail4'] * np.exp(data['U_4'] / beta[6])
                                    + data['avail5'] * np.exp(data['U_5'] / beta[6])
                                    + (1 - data['avail4']) * (1 - data['avail5']))
    
    # Nest probabilities
    data['P_nest_1'] = np.exp(beta[5] * data['logsum_1']) / \
                                 (np.exp(beta[5] * data['logsum_1']) 
                                  + np.exp(beta[6] * data['logsum_2']))
    data['P_nest_2'] = 1 - data['P_nest_1']
    
    # Within nest probabilities for nests with > 1 alt
    data['P_1_in_nest1'] = data['avail1'] * np.exp(data['U_1'] / beta[5]) / \
                                (data['avail1'] * np.exp(data['U_1'] / beta[5]) 
                                 + data['avail2'] * np.exp(data['U_2'] / beta[5])
                                 + data['avail3'] * np.exp(data['U_3'] / beta[5]))
    data['P_2_in_nest1'] = data['avail2'] * np.exp(data['U_2'] / beta[5]) / \
                                (data['avail1'] * np.exp(data['U_1'] / beta[5]) 
                                 + data['avail2'] * np.exp(data['U_2'] / beta[5])
                                 + data['avail3'] * np.exp(data['U_3'] / beta[5]))
    data['P_3_in_nest1'] = 1 - data['P_1_in_nest1'] - data['P_2_in_nest1']
    
    data['P_4_in_nest2'] = data['avail4'] * np.exp(data['U_4'] / beta[6]) / \
                            (data['avail4'] * np.exp(data['U_4'] / beta[6])
                                + data['avail5'] * np.exp(data['U_5'] / beta[6]))
    data['P_4_in_nest2'] = data['P_4_in_nest2'].fillna(0)
    data['P_5_in_nest2'] = 1 - data['P_4_in_nest2']
    
    # Full probabilities
    data['P_1'] = data['P_nest_1'] * data['P_1_in_nest1']
    data['P_2'] = data['P_nest_1'] * data['P_2_in_nest1']
    data['P_3'] = data['P_nest_1'] * data['P_3_in_nest1']
    data['P_4'] = data['P_nest_2'] * data['P_4_in_nest2']
    data['P_5'] = data['P_nest_2'] * data['P_5_in_nest2']
    
    # Calculate probability for chosen alternative for each row
    data['P'] = (data['choice'] == 1) * data['P_1'] + \
                (data['choice'] == 2) * data['P_2'] + \
                (data['choice'] == 3) * data['P_3'] + \
                (data['choice'] == 4) * data['P_4'] + \
                (data['choice'] == 5) * data['P_5']
    
    # Replace zero probabilities with small value to avoid LL = -inf
    epsilon = 1e-20
    data.loc[data['P'] == 0, 'P'] = epsilon
    
    # Calculate log-likelihood 
    LL = data['P'].apply(np.log).sum()
    
    return -LL  # We minimize negative log-likelihood