""":file: MEV_telephone_cross_var.py

Version for Biogeme 3.2.13

:author: Michel Bierlaire
:date: Fri Apr 28 10:19:38 2023

"""

import biogeme.results as res
from biogeme.exceptions import BiogemeError
import biogeme.biogeme as bio
from biogeme.expressions import Beta
from biogeme.models import logcnl
from biogeme.nests import OneNestForCrossNestedLogit, NestsForCrossNestedLogit

from telephone_data import (
    database,
    choice,
    avail1,
    avail2,
    avail3,
    avail4,
    avail5,
    logcostBM,
    logcostSM,
    logcostLF,
    logcostEF,
    logcostMF,
)

# Parameters to be estimated
# Arguments:
#   1  Name for report. Typically, the same as the variable
#   2  Starting value
#   3  Lower bound
#   4  Upper bound
#   5  0: estimate the parameter, 1: keep it fixed

ASC_BM = Beta('ASC_BM', 0, None, None, 0)
ASC_EF = Beta('ASC_EF', 0, None, None, 0)
ASC_LF = Beta('ASC_LF', 0, None, None, 0)
ASC_MF = Beta('ASC_MF', 0, None, None, 0)
B_COST = Beta('B_COST', 0, None, None, 0)

# Nest parameters

N_FLAT = Beta('N_FLAT', 1, None, None, 0)
N_MEAS = Beta('N_MEAS', 1, None, None, 0)

a_FLAT_LF = Beta('a_FLAT_LF', 0.5, 0, 1, 0)
a_MEAS_LF = 1 - a_FLAT_LF

# Utilities

V_BM = ASC_BM + B_COST * logcostBM
V_SM = B_COST * logcostSM
V_LF = ASC_LF + B_COST * logcostLF
V_EF = ASC_EF + B_COST * logcostEF
V_MF = ASC_MF + B_COST * logcostMF

V = {1: V_BM, 2: V_SM, 3: V_LF, 4: V_EF, 5: V_MF}
avail = {1: avail1, 2: avail2, 3: avail3, 4: avail4, 5: avail5}

# Definitions of nests

alpha_N_FLAT = {1: 0, 2: 0, 3: a_FLAT_LF, 4: 1, 5: 1}
alpha_N_MEAS = {1: 1, 2: 1, 3: a_MEAS_LF, 4: 0, 5: 0}

nest_N_FLAT = OneNestForCrossNestedLogit(
    nest_param=N_FLAT, dict_of_alpha=alpha_N_FLAT, name='flat'
)
nest_N_MEAS = OneNestForCrossNestedLogit(
    nest_param=N_MEAS, dict_of_alpha=alpha_N_MEAS, name='measured'
)

nests = NestsForCrossNestedLogit(
    choice_set=list(V), tuple_of_nests=(nest_N_FLAT, nest_N_MEAS)
)

# CNL with fixed alphas

logprob = logcnl(V, avail, nests, choice)
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'MEV_telephone_cross_var'
the_biogeme.calculateNullLoglikelihood(avail)
results = the_biogeme.estimate()

# Get the results in a pandas table

pandasResults = results.getEstimatedParameters()
print(pandasResults)

pandasResults

print(f'Nbr of observations: {database.getNumberOfObservations()}')
print(f'LL(0) =    {results.data.initLogLike:.3f}')
print(f'LL(beta) = {results.data.logLike:.3f}')
print(f'AIC = {results.data.akaike:.1f}')
print(f'Output file: {results.data.htmlFileName}')

# Likelihood ratio test
# Compare with the nested logit model

try:
    nested_results = res.bioResults(pickleFile='MEV_telephone_nested.pickle')
    ll_nested = nested_results.data.logLike
    aic_nested = nested_results.data.akaike
    ll_cross_var = results.data.logLike
    aic_cross_var = results.data.akaike

    print(
        f'LL nested:  {ll_nested:.3f}  '
        f'AIC: {aic_nested:.3f}  '
        f'Parameters: {nested_results.data.nparam}'
    )
    print(
        f'LL CNL (var): {ll_cross_var:.3f}  '
        f'AIC: {aic_cross_var:.3f}  '
        f'Parameters: {results.data.nparam}'
    )

    lr_test = results.likelihood_ratio_test(nested_results)
    print(f'Likelihood ratio: {lr_test.statistic:.3f}')
    print(f'Test threshold: {lr_test.threshold:.3f}')
    print(f'Test diagnostic: {lr_test.message}')

except BiogemeError:
    print('To perform the test, run first the script MEV_telephone_nested.py')