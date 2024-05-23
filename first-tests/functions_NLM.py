import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t

def estimate_nested_logit(data, beta_initial, beta_names, log_likelihood_function):
    """
    Estimate parameters for a nested logit model using maximum likelihood estimation.

    Args:
    - data (DataFrame): Input dataset containing variables needed for the model.
    - beta_initial (array-like): Initial guess for model parameters.
    - beta_names (list): Names of model parameters.
    - log_likelihood_function (function): Function that calculates the log-likelihood of the model. 

    Returns:
    - result (OptimizeResult): Result object from scipy.optimize.minimize containing optimization results.
    - se (array-like): Robust asymptotic standard errors of parameter estimates.
    - t_stat (array-like): t-statistics of parameter estimates.
    - p_value (array-like): p-values of parameter estimates.
    """

    # Run the model
    result = minimize(log_likelihood_function, x0 = beta_initial, args = data, method='BFGS')

    # Calculate Hessian matrix
    hessian_inv = result.hess_inv

    # Calculate robust asymptotic standard errors
    se = np.sqrt(np.diag(hessian_inv))

    # Calculate t-statistics
    t_stat = result.x / se

    # Calculate p-values
    p_value = (1 - t.cdf(np.abs(t_stat), len(data) - len(beta_initial))) * 2

    # Calculate AIC
    log_likelihood_value = -result.fun
    k = len(beta_initial)
    n = len(data)
    aic = 2 * k - 2 * log_likelihood_value

    # Calculate BIC
    bic = np.log(n) * k - 2 * log_likelihood_value

    # Create DataFrame to store results
    results_df = pd.DataFrame({
        "Parameter": beta_names,
        "Estimate": result.x,
        "Robust Asymptotic SE": se,
        "t-statistic": t_stat,
        "p-value": p_value
    })

    print("Optimization Results:")
    print(results_df)
    print("AIC:", aic)
    print("BIC:", bic)
    print("Log-likelihood:", -result.fun)

    return result, se, t_stat, p_value, aic, bic

def simulate_choice(row):
    """
    Function to simulate a choice based on cumulative probabilities.

    Parameters:
    - row: pandas Series or DataFrame row containing probabilities for each choice

    Returns:
    - The simulated choice (an integer between 1 and the number of choices)
    """
    # Generate a random number between 0 and 1
    random_number = np.random.rand()
    
    # Calculate cumulative probabilities
    probabilities = row[['P_1', 'P_2', 'P_3', 'P_4', 'P_5']]
    cumulative_probs = probabilities.cumsum()
    
    # Determine simulated choice based on random number
    for i, cumulative_prob in enumerate(cumulative_probs):
        if random_number <= cumulative_prob:
            return i+1
        
def multivariate_lognormal_pdf(x, mu, sigma):
    """
    Compute the probability density function (PDF) of a multivariate lognormal distribution.

    Parameters:
        x (ndarray): Vector in log space.
        mu (ndarray): Mean vector.
        sigma (ndarray): Covariance matrix.

    Returns:
        float: Value of the PDF at x.
    """
    n = len(x)
    
    # Compute the PDF of a multivariate normal distribution
    exponent = -0.5 * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu)
    mvn_pdf = (1 / ((2 * np.pi) ** (n / 2) * np.sqrt(np.linalg.det(sigma)))) * np.exp(exponent)
    
    # Compute the Jacobian determinant term and remove x = 0 elements
    jacobian = np.prod(1 / x)

    pdf = jacobian * mvn_pdf

    # Compute the PDF of the multivariate lognormal distribution
    
    return pdf

def largest_cluster_size(q_t_given_x):
    """
    Calculate the size of the largest cluster in a given set of clusters.

    Parameters:
    - q_t_given_x (numpy.ndarray): A 2D array representing the clusters,
                                where each row represents a data point and each column represents a cluster.

    Returns:
    - int: The size of the largest cluster.

    """
    # Calculate the size of each cluster
    cluster_sizes = np.sum(q_t_given_x, axis=0)
    # Find the size of the largest cluster
    largest_cluster_size = np.max(cluster_sizes)
    return largest_cluster_size
