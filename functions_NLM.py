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

    return result, se, t_stat, p_value, aic, bic
