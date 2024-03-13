""":file: MEV_telephone_nested.py

Version for Biogeme 3.2.13

:author: Michel Bierlaire
:date: Fri Apr 28 10:20:52 2023

"""

import biogeme.biogeme as bio
from biogeme.expressions import Beta
from biogeme.models import loglogit, lognested
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit

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

N_FLAT = Beta('N_FLAT', 1, 1, None, 0)
N_MEAS = Beta('N_MEAS', 1, 1, None, 0)

# Utilities

V_BM = ASC_BM + B_COST * logcostBM
V_SM = B_COST * logcostSM
V_LF = ASC_LF + B_COST * logcostLF
V_EF = ASC_EF + B_COST * logcostEF
V_MF = ASC_MF + B_COST * logcostMF

V = {1: V_BM, 2: V_SM, 3: V_LF, 4: V_EF, 5: V_MF}
avail = {1: avail1, 2: avail2, 3: avail3, 4: avail4, 5: avail5}

# Definitions of nests

N_MEAS = OneNestForNestedLogit(
    nest_param=N_MEAS, list_of_alternatives=[1, 2], name='measured'
)
N_FLAT = OneNestForNestedLogit(
    nest_param=N_FLAT, list_of_alternatives=[3, 4, 5], name='flat'
)

nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(N_FLAT, N_MEAS))


# Nested logit

logprob = lognested(V, avail, nests, choice)
the_biogeme = bio.BIOGEME(database, logprob)
the_biogeme.modelName = 'MEV_telephone_nested'
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


# Compare with the logit model

logprob_logit = loglogit(V, avail, choice)
biogeme_logit = bio.BIOGEME(database, logprob_logit)
biogeme_logit.modelName = 'MEV_telephone_logit'
biogeme_logit.calculateNullLoglikelihood(avail)
results_logit = biogeme_logit.estimate()

ll_logit = results_logit.data.logLike
aic_logit = results_logit.data.akaike
ll_nested = results.data.logLike
aic_nested = results.data.akaike

print(
    f'LL logit:  {ll_logit:.3f}  '
    f'AIC: {aic_logit:.3f}  '
    f'Parameters: {results_logit.data.nparam}'
)
print(
    f'LL nested: {ll_nested:.3f}  '
    f'AIC: {aic_nested:.3f}  '
    f'Parameters: {results.data.nparam}'
)

lr_test = results.likelihood_ratio_test(results_logit)
print(f'Likelihood ratio: {lr_test.statistic:.3f}')
print(f'Test threshold: {lr_test.threshold:.3f}')
print(f'Test diagnostic: {lr_test.message}')
