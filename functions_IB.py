import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler divergence between two discrete probability distributions.

    Parameters:
    - p (array-like): The first probability distribution.
    - q (array-like): The second probability distribution.

    Returns:
    - float: The KL divergence between distributions p and q.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Ensure that distributions sum up to 1
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Ignore division by zero warning when q[i] is 0
    with np.errstate(divide='ignore'):
        kl = np.sum(p * np.log(p / q))
    
    return kl

def entropy(probabilities):
    """
    Calculate the entropy of a discrete probability distribution.

    Parameters:
    - probabilities (array-like): Array-like object containing the probabilities.

    Returns:
    - float: Entropy value.
    """
    # Convert probabilities to a NumPy array if necessary
    probabilities = np.array(probabilities)
    
    # Ensure probabilities sum up to 1
    assert np.isclose(np.sum(probabilities), 1.0), "Probabilities must sum up to 1"
    
    # Remove zeros to avoid issues with log calculation
    probabilities = probabilities[probabilities > 0]
    
    # Calculate entropy
    return -np.sum(probabilities * np.log(probabilities))

def mutual_information(p_xy):
    """
    Compute the mutual information between two random variables X and Y given their joint probability distribution.

    Parameters:
    - p_xy (numpy.ndarray): Joint probability distribution of X and Y.

    Returns:
    - float: Mutual information between X and Y.
    """
    # Compute marginal distributions
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    # Compute entropy
    entropy_x = -np.sum(p_x * np.log(p_x + 1e-10))
    entropy_y = -np.sum(p_y * np.log(p_y + 1e-10))
    joint_entropy = -np.sum(p_xy * np.log(p_xy + 1e-10))  # Adding a small value to avoid log(0)
    
    # Compute mutual information
    mutual_info_x_y = entropy_x + entropy_y - joint_entropy
    return mutual_info_x_y

def generate_joint_distribution(n, m):
    """
    Generate a joint probability distribution matrix.

    Parameters:
    - n (int): The number of rows in the matrix.
    - m (int): The number of columns in the matrix.

    Returns:
    - numpy.ndarray: The generated joint probability distribution matrix.

    """
    # Generate random probabilities for each combination of n and m
    joint_prob = np.random.rand(n, m)
    
    # Normalize each row to ensure they sum up to 1
    joint_prob /= np.sum(joint_prob, keepdims=True)
    
    return joint_prob

def information_bottleneck(p_xy, beta, max_iter=100):
    """
    Compute the information bottleneck algorithm.

    Parameters:
    - p_xy (numpy.ndarray): The joint distribution of X and Y. Each row is an X, each column is a Y.
    - beta (float): The trade-off parameter controlling the amount of compression.
    - max_iter (int): The maximum number of iterations for the algorithm. Default is 100. 

    Returns:
    - q_t_given_x (numpy.ndarray): The posterior distribution of T given X.
    - q_t (numpy.ndarray): The posterior distribution of T.
    - q_y_given_t (numpy.ndarray): The posterior distribution of Y given T.

    Remarks:
    when we have p(a|b), a is the column and b is the row. 
    """
    # Each row is an X, each column is a Y
    num_data_points = p_xy.shape[0]
    num_clusters = p_xy.shape[0]

    #Initialize p(y|x) and p(x)
    p_x = np.sum(p_xy, axis=1)
    p_y_given_x = p_xy / np.sum(p_xy, axis=1, keepdims=True) # Each row is an X, each column is a Y
    
    # Initialize q(t|x), q(t), and q(y|t)
    q_t_given_x = np.random.rand(num_data_points, num_clusters) # Each row is an X, each column is a T
    q_t_given_x /= np.sum(q_t_given_x, axis=1, keepdims=True) # We have to normalize the rows so that they sum to 1

    q_t = np.dot(p_x, q_t_given_x) # Array of dimension Y 
    q_y_given_t = np.dot(p_xy.T, q_t_given_x) / q_t
    q_y_given_t = q_y_given_t.T # Each row is a T, each column is a Y
    
    for _ in range(max_iter):
        # Compute KL divergence and update q(t|x)
        for i in range(num_data_points):
            for j in range(num_clusters):
                d_xt = kl_divergence(p_y_given_x[i], q_y_given_t[j])
                q_t_given_x[i, j] = q_t[j] * np.exp(-beta * d_xt)
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        q_t_given_x += epsilon
        q_t_given_x /= np.sum(q_t_given_x, axis=1, keepdims=True)

        # Update q(t) and q(y|t)
        q_t = np.dot(p_x, q_t_given_x) # Array of dimension Y 
        q_y_given_t = np.dot(p_xy.T, q_t_given_x) / q_t
        q_y_given_t = q_y_given_t.T
        
        # print the iteration number
        print(f'Iteration {_}', "out of", max_iter)
    return q_t_given_x, q_t, q_y_given_t

def information_bottleneck_convergence(p_xy, beta, max_iter=10000, threshold=1e-8):

    """
    Compute the information bottleneck algorithm with convergence based on the diference between 
    consecutive values of q(t).

    Parameters:
    - p_xy (numpy.ndarray): The joint distribution of X and Y. Each row is an X, each column is a Y.
    - beta (float): The trade-off parameter controlling the amount of compression.
    - max_iter (int): The maximum number of iterations for convergence. Default is 10000.
    - threshold (float): The convergence threshold. Default is 1e-8.

    Returns:
    - q_t_given_x (numpy.ndarray): The posterior distribution of T given X.
    - q_t (numpy.ndarray): The posterior distribution of T.
    - q_y_given_t (numpy.ndarray): The posterior distribution of Y given T.

    Remarks:
    when we have p(a|b), a is the column and b is the row. 
    """
    # Each row is an X, each column is a Y
    num_data_points = p_xy.shape[0]
    #num_clusters = p_xy.shape[1]
    num_clusters = p_xy.shape[0]

    #Initialize p(y|x) and p(x)
    p_x = np.sum(p_xy, axis=1)
    p_y_given_x = p_xy / np.sum(p_xy, axis=1, keepdims=True) # Each row is an X, each column is a Y
    
    # Initialize q(t|x), q(t), and q(y|t)
    q_t_given_x = np.random.rand(num_data_points, num_clusters) # Each row is an X, each column is a T
    q_t_given_x /= np.sum(q_t_given_x, axis=1, keepdims=True) # We have to normalize the rows so that they sum to 1

    q_t = np.dot(p_x, q_t_given_x) # Array of dimension Y 
    q_y_given_t = np.dot(p_xy.T, q_t_given_x) / q_t
    q_y_given_t = q_y_given_t.T # Each row is a T, each column is a Y

    q_t_old = np.zeros(num_clusters)
    q_t_new = q_t
    iteration = 0
    
    while (np.linalg.norm(q_t_old - q_t_new) > threshold and iteration < max_iter):
        iteration += 1
        # print iteration number 
        print(f'Iteration {iteration}')

        # Compute KL divergence and update q(t|x)
        for i in range(num_data_points):
            for j in range(num_clusters):
                d_xt = kl_divergence(p_y_given_x[i], q_y_given_t[j])
                q_t_given_x[i, j] = q_t[j] * np.exp(-beta * d_xt)
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        q_t_given_x += epsilon
        q_t_given_x /= np.sum(q_t_given_x, axis=1, keepdims=True)

        # Update q(t) and q(y|t)
        q_t_old = q_t
        q_t = np.dot(p_x, q_t_given_x) # Array of dimension Y 
        q_t_new = q_t
        q_y_given_t = np.dot(p_xy.T, q_t_given_x) / q_t
        q_y_given_t = q_y_given_t.T
    return q_t_given_x, q_t, q_y_given_t

def compute_mutual_information_over_beta(p_xy, beta_values, max_iter, algorithm):
    """
    Compute mutual information between variables X and T, and between variables T and Y
    for different values of beta.

    Parameters:
    - p_xy: numpy array, joint probability distribution of variables X and Y
    - beta_values: list, values of beta for which to compute mutual information
    - max_iter: int, maximum number of iterations for the information bottleneck algorithm
    - algorithm: function, the information bottleneck algorithm to use

    Returns:
    - mutual_info_x_t_list: list, mutual information between X and T for each beta value
    - mutual_info_t_y_list: list, mutual information between T and Y for each beta value
    """
    mutual_info_x_t_list = []
    mutual_info_t_y_list = []

    p_x = np.sum(p_xy, axis=1)
    
    beta_n = 0
    for beta in beta_values:
        #print the beta number
        beta_n += 1
        print(f'Computing mutual information for beta {beta_n} out of {len(beta_values)}')
        
        q_t_given_x, q_t, q_y_given_t = algorithm(p_xy, beta=beta, max_iter=max_iter)
        
        # Compute mutual information
        q_xt = p_x[:, np.newaxis] * q_t_given_x
        q_ty = q_t[:, np.newaxis] * q_y_given_t
        mutual_info_x_t = mutual_information(q_xt)
        mutual_info_t_y = mutual_information(q_ty)
        
        mutual_info_x_t_list.append(mutual_info_x_t)
        mutual_info_t_y_list.append(mutual_info_t_y)
    
    return mutual_info_x_t_list, mutual_info_t_y_list

def IB_curve(p_xy, beta_values, max_iter, algorithm):
    """
    Plot the IB curve for a given joint distribution p_xy.

    Parameters:
    - p_xy (numpy.ndarray): The joint distribution p(x, y) as a 2D numpy array.
    - beta_values (list): The list of beta values to compute mutual information over.
    - max_iter (int): The maximum number of iterations for the information bottleneck algorithm.
    - algorithm (str): The algorithm to use for computing mutual information.

    Returns:
    - The IB curve plot.

    """
    # Compute mutual information over beta values
    mutual_info_x_t_list, mutual_info_t_y_list = compute_mutual_information_over_beta(p_xy, beta_values, max_iter, algorithm)

    # Compute mutual information from the joint distribution
    mutual_info_xy = mutual_information(p_xy)

    #Compute log(|T|) where |T| is at most |X|
    log_T = np.log(len(p_xy[:, 0])) 

    # Plot I(T,Y) against I(X,T)
    plt.plot(mutual_info_x_t_list, mutual_info_t_y_list, marker='o')
    # Add horizontal line at the value of I(X,Y)
    plt.axhline(y=mutual_info_xy, color='r', linestyle='--', label='I(X;Y)')
    # Add vertical line at the value of H(T)
    plt.axvline(x=log_T, color='g', linestyle='--', label='log(|T|)')

    plt.xlabel('I(X, T)')
    plt.ylabel('I(T, Y)')
    plt.title('IB curve')
    plt.legend()
    plt.show()
