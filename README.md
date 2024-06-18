# Application of the information-theoretic clustering to the discrete choice

## Thesis description

This thesis investigates the application of the Deterministic Information Bottleneck (DIB) algorithm to discrete choice models, aiming to establish a link between these two methodologies to enhance model selection and evaluation. Discrete choice models are essential for understanding decision-making processes, yet they often face challenges in model selection due to high-dimensional data and complex preference structures. The DIB algorithm, which optimizes the trade-off between relevance and compression of information, offers a promising approach to address these issues.

In this study, we demonstrate that the DIB algorithm can identify the best-fitting discrete choice model by maximizing log-likelihood and minimizing the Akaike Information Criterion (AIC). By integrating DIB with discrete choice theory, we develop a novel framework for model selection that leverages information-theoretic principles to improve predictive performance and interpretability.

To validate our approach, we apply the proposed method to different datasets. The empirical results show that models selected using the DIB algorithm consistently outperform other possible models in terms of log-likelihood and AIC. These findings support our conjecture that DIB can serve as a powerful tool for model selection in discrete choice analysis, providing more accurate and reliable insights into consumer behavior.

Overall, this research contributes to the field of discrete choice modelling by introducing an information- theoretic perspective, which enhances both the theoretical understanding and practical application of these models.

## Description of the files 

This repository contains all the files used to test and generate the results and plots in the thesis report. Here are an exhaustive list of the files that you find in the repository. 

- data (folder): contains different datasets. 

- py-files (folder): contains all the .py files used in the different notebooks. Here is an exhaustive list of the files : 

    - functions_general_IB.py: File containing different general functions and specific functions for the IB algorithm. Here is an exhaustive list: 

        - kl_divergence: Compute the D_KL(p,q) between two distribution functions p and q.
    
        - entropy: Compute H(p) of a distribution function p.

        - mutual_information: Compute I(X,Y) from their joint distribution p(x,y).

        - generate_joint_distribution: Generate a valid joint distribution p(x,y) of chosen size.

        - information_bottleneck: Compute q(t|x), q(t) and q(y|t) from a joint distribution p(x,y) by using the information bottleneck method (soft clustering) for a fixed number of iterations. 

        - information_bottleneck_convergence: Compute q(t|x), q(t) and q(y|t) from a joint distribution p(x,y) by using the information bottleneck method (soft clustering) without fixing the number of iterations but by using a metric criterion for convergence. 

        - compute_mutual_information_over_beta: Compute I(X;T) and I(T;Y) for different chosen values of beta from the joint distribution p(x,y). We can choose which algorithm we want to use in this function. 
    
        - IB_curve: Compute an IB curve from a joint distribution p(x,y) and some values of beta. 

    - functions_geom_DIB.py: File containing different functions specific to the DIB algorithm for geometric clustering. Here is an exhaustive list : 
    
        - generate_gaussian_points: Generate Gaussian points from multiple chosen Gaussian distributions.

        - add_index_to_data: Create a dataframe of datapoints where each row is a data location and the first column is the data index. 

        - px_i: Compute p(x|i) following the rules for geometric clustering, cf. [3].

        - calculate_probabilities: Compute the probabilities p(x|i) and p(i, x) for each data point in the given DataFrame.

        - geom_DIB: Compute q(t|x), q(t) and q(y|t) from a joint distribution p(x,y) by using the deterministic information bottleneck method (hard clustering) for a fixed number of iterations. 

        - plot_clusters: Plot the data points and color them based on the cluster they are associated with (q(t|x)).

        - DIB_curve: Plot the DIB curve from a joint distribution p(x,y) and some values of beta by using geom_DIB. 

        - compute_entropy_over_beta: Compute H(T) for different chosen values of beta from the joint distribution p(x,y). We can choose which algorithm we want to use in this function. 

        - geom_DIB_on_alternatives: Compute q(t|x), q(t) and q(y|t) from a joint distribution p(x,y) by using the deterministic information bottleneck method (hard clustering) without fixing the number of iterations but by using a metric criterion for convergence. It also fixes the maximum number of clusters to the number of alternatives instead of the number of datapoints.

        - DIB_curve_on_alternatives: Plot the DIB curve from a joint distribution p(x,y) and some values of beta by using geom_DIB_on_alternatives.

    - functions_NLM.py: File containing different functions to compute a NLM. Here is an exhaustive list:

        - estimate_nested_logit: Estimate parameters for a nested logit model using maximum likelihood estimation.

        - simulate_choice: Function to simulate a choice based on cumulative probabilities.

        - multivariate_lognormal_pdf: Compute the probability density function (PDF) of a multivariate lognormal distribution.

        - largest_cluster_size(q_t_given_x): Calculate the size of the largest cluster in a given set of clusters.

    - archive-telephone-likelihood.py: Backup file containing different likelihood functions for telephone dataset. 

- first-tests (folder): contains all the notebooks used to test IB framework at the very beginning of the project.

- notebook-airline (folder): contains all the notebooks used to test IB framework on the Airline dataset.

- notebook-optima (folder): contains all the notebooks used to test IB framework on the Optima dataset.

- notebook-optima-2 (folder): contains all the notebooks used to test IB framework on the Optima dataset with a different base model.

- notebook-telephone (folder): contains all the notebooks used to test IB framework on the telephone dataset.

- notebook-SM (folder): contains all the notebooks used to test IB framework on the SM dataset. 
