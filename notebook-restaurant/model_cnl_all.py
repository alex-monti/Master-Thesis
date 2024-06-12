"""Script containing the model specification

Michel Bierlaire
Wed Nov  1 17:37:33 2023
"""
from biogeme.expressions import Beta, Variable, log
# from biogeme.sampling_of_alternatives import CrossVariableTuple

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models

import pandas as pd
# import numpy as np

asian_nest_ids = [0,
 1,
 3,
 13,
 15,
 17,
 18,
 27,
 31,
 33,
 34,
 37,
 40,
 45,
 47,
 50,
 51,
 55,
 57,
 68,
 70,
 72,
 76,
 78,
 79,
 80,
 81,
 87,
 89,
 91,
 92,
 94,
 98]

nonasian_nest_ids = [2,
 4,
 5,
 6,
 7,
 8,
 9,
 10,
 11,
 12,
 14,
 16,
 19,
 20,
 21,
 22,
 23,
 24,
 25,
 26,
 28,
 29,
 30,
 32,
 35,
 36,
 38,
 39,
 41,
 42,
 43,
 44,
 46,
 48,
 49,
 52,
 53,
 54,
 56,
 58,
 59,
 60,
 61,
 62,
 63,
 64,
 65,
 66,
 67,
 69,
 71,
 73,
 74,
 75,
 77,
 82,
 83,
 84,
 85,
 86,
 88,
 90,
 93,
 95,
 96,
 97,
 99]

alternatives = pd.read_csv('restaurants.dat')
alternatives['ID'] = alternatives['ID'].astype(int)

alternatives['alpha_asian'] = (alternatives['Asian']==1)/(alternatives['Asian']+alternatives['downtown'])
alternatives['alpha_downtown'] = (alternatives['downtown']==1)/(alternatives['Asian']+alternatives['downtown'])
alternatives['alpha_other'] = (alternatives['downtown']==0)*(alternatives['Asian']==0)*1

alternatives['alpha_asian'] = alternatives['alpha_asian'].fillna(0)
alternatives['alpha_downtown'] = alternatives['alpha_downtown'].fillna(0)

observations = pd.read_csv('obs_choice.dat')

reshaped_data = {}

# Iterating over each row and renaming the columns
for i, row in alternatives.iterrows():
    for col in alternatives.columns:
        # Renaming each column and adding its value to the reshaped_data dictionary
        reshaped_data[f'{col}_{i}'] = row[col]

# Converting the dictionary to a DataFrame with a single row
reshaped_df = pd.DataFrame([reshaped_data])
repeated_df = pd.concat([reshaped_df] * len(observations), ignore_index=True)

data = pd.concat([observations, repeated_df], axis=1)

database = db.Database('restaurants', data)

del(observations)
del(repeated_df)
del(reshaped_df)
del(reshaped_data)

alpha_downtown_dict = alternatives['alpha_downtown'].to_dict()
alpha_asian_dict = alternatives['alpha_asian'].to_dict()
alpha_other_dict = alternatives['alpha_other'].to_dict()


# combined_variables = [
#     CrossVariableTuple(
#         'log_dist',
#         log(
#             (
#                 (Variable('user_lat') - Variable('rest_lat')) ** 2
#                 + (Variable('user_lon') - Variable('rest_lon')) ** 2
#             )
#             ** 0.5
#         ),
#     )
# ]

# Parameters to estimate
beta_rating = Beta('beta_rating', 0, None, None, 0)
beta_price = Beta('beta_price', 0, None, None, 0)
beta_chinese = Beta('beta_chinese', 0, None, None, 0)
beta_japanese = Beta('beta_japanese', 0, None, None, 0)
beta_korean = Beta('beta_korean', 0, None, None, 0)
beta_indian = Beta('beta_indian', 0, None, None, 0)
beta_french = Beta('beta_french', 0, None, None, 0)
beta_mexican = Beta('beta_mexican', 0, None, None, 0)
beta_lebanese = Beta('beta_lebanese', 0, None, None, 0)
beta_ethiopian = Beta('beta_ethiopian', 0, None, None, 0)
beta_log_dist = Beta('beta_log_dist', 0, None, None, 0)

mu_asian = Beta('mu_asian', 1.0, 1.0, None, 0)
mu_downtown = Beta('mu_downtown', 1, 1, None, 0)

V = {i:(
    beta_rating * Variable(f'rating_{i}')
    + beta_price * Variable(f'price_{i}')
    + beta_chinese * Variable(f'category_Chinese_{i}')
    + beta_japanese * Variable(f'category_Japanese_{i}')
    + beta_korean * Variable(f'category_Korean_{i}')
    + beta_indian * Variable(f'category_Indian_{i}')
    + beta_french * Variable(f'category_French_{i}')
    + beta_mexican * Variable(f'category_Mexican_{i}')
    + beta_lebanese * Variable(f'category_Lebanese_{i}')
    + beta_ethiopian * Variable(f'category_Ethiopian_{i}')
    + beta_log_dist * (log(
        (
            (Variable('user_lat') - Variable(f'rest_lat_{i}')) ** 2
            + (Variable('user_lon') - Variable(f'rest_lon_{i}')) ** 2
        )
        ** 0.5
    ))
)
for i in range(100)
     }

asian = mu_asian, alpha_asian_dict
downtown = mu_downtown, alpha_downtown_dict

nests = asian, downtown

ii = 1

CHOICE = Variable(f'cnl_{ii}')

logprob_cnl = models.logcnl_avail(V, None, nests, CHOICE)
biogeme = bio.BIOGEME(database, logprob_cnl)
biogeme.modelName = f'cnl_restaursnts_{ii}'
nested_existing_results = biogeme.estimate(recycle=False)

