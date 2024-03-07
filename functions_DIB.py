import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from functions import kl_divergence, entropy, mutual_information

def generate_gaussian_points(num_points_per_distribution, mean_list, cov_list):
    """
    Generate Gaussian points from multiple distributions.

    Parameters:
    - num_points_per_distribution (int): Number of points to generate per distribution.
    - mean_list (list): List of mean vectors for each distribution.
    - cov_list (list): List of covariance matrices for each distribution.

    Returns:
    - points (ndarray): Array of generated points from all distributions.
    """
    points = []
    for mean, cov in zip(mean_list, cov_list):
        points.extend(np.random.multivariate_normal(mean, cov, num_points_per_distribution))
    return np.array(points)

def add_index_to_data(data): 
    """
    Adds an index column to the given data.

    Parameters:
    - data (list): A list of tuples representing the data points.

    Returns:
    - pandas.DataFrame: A DataFrame with an additional 'Index' column.

    """
    df = pd.DataFrame(data, columns=['X', 'Y'])
    df['Index'] = df.index
    return df[['Index', 'X', 'Y']]

def px_i(data_point, data_point_i, s):
    """
    Calculate p(x|i) for a given data point.

    Parameters:
    - data_point (numpy.ndarray): The data point for which the probability density function value is calculated.
    - data_point_i (numpy.ndarray): The data point index.
    - s (float): The unit of distance.

    Returns:
    - float: The probability density function value for the given data point.
    """
    proba = np.exp(-1/(2*s**2) * np.linalg.norm(data_point - data_point_i)**2 )
    return proba

def calculate_probabilities(df, s=1):
    """
    Calculate the probabilities p(x|i) and p(i, x) for each data point in the given DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data points.
    - s (float): The unit of distance for the p(x|i) calculation. Default is 1.

    Returns:
    - px_given_i (numpy.ndarray): The probabilities p(x|i) for each data point.
    - joint_pix (numpy.ndarray): The joint probabilities p(i, x) for each data point.
    """
    n_data = len(df)
    px_given_i = []
    joint_pix = []
    for i in range(len(df)):
        data_point = df[['X', 'Y']].iloc[i].values
        index = df['Index'].iloc[i]
        
        # Calculate p(x|i)
        px_i_values = [px_i(data_point, data_point_i, s) for data_point_i in df[['X', 'Y']].values]
        # Normalize p(x|i) to ensure sum equals 1
        px_i_values /= np.sum(px_i_values)
        px_given_i.append(px_i_values)
        
        # Calculate p(i, x)
        p_i = 1/n_data
        pix_values = px_i_values * p_i
        joint_pix.append(pix_values)
    
    px_given_i = np.array(px_given_i)
    joint_pix = np.array(joint_pix)
    return px_given_i, joint_pix

def geom_DIB(p_xy, max_iter=100, beta=0.5):
    """
    Performs the geometric information bottleneck algorithm for clustering.

    Parameters:
    - p_xy (numpy.ndarray): The joint probability distribution of data points and clusters.
    - max_iter (int): The maximum number of iterations for the algorithm. Default is 100.
    - beta (float): The beta parameter for the algorithm. Default is 0.5.

    Returns:
    - q_t_given_x (numpy.ndarray): The conditional probability distribution of clusters given data points.
    - q_t (numpy.ndarray): The marginal probability distribution of clusters.
    - q_y_given_t (numpy.ndarray): The conditional probability distribution of data points given clusters.
    """
    num_data_points = p_xy.shape[0]
    num_clusters = p_xy.shape[0]
    
    # Initialize f(x) as if each index i is assigned to its own cluster
    f_x = np.arange(num_data_points)

    # Initialization 
    d_xt = np.zeros((num_data_points, num_clusters))
    p_y_given_x = p_xy / np.sum(p_xy, axis=1, keepdims=True)
    p_x = np.sum(p_xy, axis=1)
    
    # Initialize q(t) and q(y|t)
    q_t = np.zeros(num_clusters)
    q_y_given_t = np.zeros_like(p_xy)
    for t in range(num_clusters):
        relevant_indices = np.where(f_x == t)[0]
        if len(relevant_indices) > 0:
            q_t[t] = np.sum(p_x[relevant_indices])
            q_y_given_t[t] = np.sum(p_xy[relevant_indices], axis=0) / np.sum(p_x[relevant_indices])
    
    q_t /= np.sum(q_t)  # Normalize q(t)

    # Iterative algorithm
    for _ in range(max_iter):
        
        # Compute d, l_beta, and f
        for i in range(num_data_points):
            for j in range(num_clusters):
                d_xt[i, j] = kl_divergence(p_y_given_x[i], q_y_given_t[j])

        l_beta_xt = np.log(q_t) - beta * d_xt
        f_x = np.argmax(l_beta_xt, axis=1)
        
        # Update q_t_given_x
        q_t_given_x = np.eye(num_clusters)[f_x]
        
        # Update q_t and q_y_given_t
        for t in range(num_clusters):
            relevant_indices = np.where(f_x == t)[0]
            if len(relevant_indices) > 0:
                q_t[t] = np.sum(p_x[relevant_indices])
                q_y_given_t[t] = np.sum(p_xy[relevant_indices], axis=0) / np.sum(p_x[relevant_indices])
        
        # Normalize q(t)
        q_t /= np.sum(q_t)

########################################################################################################

        # Merge step to verify if we are stuck in a local minimum
        best_merge = None

        # Compute the objective function with the current clusters
        H_T = entropy(q_t)
        I_TY = mutual_information(q_t.reshape(-1, 1) * q_y_given_t)
        objective = H_T - beta * I_TY

        # Compute the objective function for each pair of consecutive clusters
        for i in range(num_clusters - 1):
            # Merge clusters i and i + 1
            merged_f_x = f_x.copy()
            merged_f_x[np.where(merged_f_x == i + 1)] = i

            # Initialize merge_q_t and merge_q_y_given_t
            merged_q_t = np.zeros(num_clusters)
            merged_q_y_given_t = np.zeros_like(p_xy)

            # Update merge_q_t and merge_q_y_given_t
            for t in range(num_clusters):
                relevant_indices = np.where(merged_f_x == t)[0]
                if len(relevant_indices) > 0:
                    merged_q_t[t] = np.sum(p_x[relevant_indices])
                    merged_q_y_given_t[t] = np.sum(p_xy[relevant_indices], axis=0) / np.sum(p_x[relevant_indices])
            
            # Compute the objective function with the merged clusters
            merged_H_T = entropy(merged_q_t)
            merged_I_TY = mutual_information(merged_q_t.reshape(-1, 1) * merged_q_y_given_t)
            merged_objective = merged_H_T - beta * merged_I_TY

            if merged_objective < objective:
                objective = merged_objective
                best_merge = i

        if best_merge is not None:
            # If the objective function can be improved, merge the clusters
            f_x[np.where(f_x == best_merge + 1)] = best_merge
            q_t_given_x = np.eye(num_clusters)[f_x]
        
            # Update q_t and q_y_given_t after the merge
            for t in range(num_clusters):
                relevant_indices = np.where(f_x == t)[0]
                if len(relevant_indices) > 0:
                    q_t[t] = np.sum(p_x[relevant_indices])
                    q_y_given_t[t] = np.sum(p_xy[relevant_indices], axis=0) / np.sum(p_x[relevant_indices])
            # Normalize q(t) 
            q_t /= np.sum(q_t)   

    return q_t_given_x, q_t, q_y_given_t

def plot_clusters(data_points, q_t_given_x):
    """
    Plot the data points and color them based on the cluster they are associated with.

    Parameters:
    - data_points (numpy.ndarray): Array of data points with shape (n_samples, n_features).
    - q_t_given_x (numpy.ndarray): Probability distribution over clusters given data points, 
                                    with shape (n_samples, n_clusters).

    Returns:
    - The plot of the data points colored by clusters.
    """
    # Determine the cluster assignment for each data point based on q(t given x)
    cluster_assignments = np.argmax(q_t_given_x, axis=1)
    
    # Define a color map for visualizing clusters
    colors = plt.cm.tab10(cluster_assignments / max(cluster_assignments))

    # Plot the data points with colors based on clusters
    plt.scatter(data_points[:, 0], data_points[:, 1], c=colors, alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data Points Colored by Clusters')
    plt.show()
