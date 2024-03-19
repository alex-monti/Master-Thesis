# Description of the files 

- images (folder): contains the different images that I saved for my report.

- data (folder): contains different datasets (telephone.dat, swissmetro.dat). 

- Telephone (folder): contains python code for different implementations of NLM and CLNM on telephone data. 

- master-project-report.pdf: report about what I learned and what is good to know about information bottleneck. 

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

    - DIB_curve: Plot the DIB curve from a joint distribution p(x,y) and some values of beta. 

    - compute_entropy_over_beta: Compute H(T) for different chosen values of beta from the joint distribution p(x,y). We can choose which algorithm we want to use in this function. 

- functions_NLM.py: File containing different functions to compute a NLM. Here is an exhaustive list: 
    - estimate_nested_logit: Estimate parameters for a nested logit model using maximum likelihood estimation.

- archive-telephone-likelihood.py: Backup file containing different likelihood functions for telephone dataset. 

- test-general-IB.ipynb: Notebook where we made the tests for functions in functions.py. 

- test-geom-DIB.ipynb: Notebook where we made the tests for functions in functions_geom_DIB.py. 

- test-general-DIB.ipynb: Notebook where we made the tests for general DIB. 

- NLM-telephone-complete-data.ipynb: Notebook where we try NLM on alt. 1,2 vs alt. 3,4,5 for telephone dataset. 

- CNLM-telephone-complete-data.ipynb: Notebook where we try CNLM on alt.1,2 vs alt. 3,4,5 for telephone dataset. 

- NLM-SM.ipynb: Notebook where we adapt an R code for NLM on swissmetro data to Python. 

- CNLM-SM.ipynb: Notebook where we adapt an R code for CNLM on swissmetro data to Python.


