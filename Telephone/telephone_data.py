"""File telephone_data.py

Data processing for the telephone case study.

:author: Michel Bierlaire, EPFL
:date: 2023-04-30 07:23:36.567625

"""

import pandas as pd
import biogeme.database as db
from biogeme.expressions import Variable, log

# Read the data

df = pd.read_csv('telephone.dat', sep='\t')
database = db.Database('telephone', df)

age0 = Variable('age0')
age1 = Variable('age1')
age2 = Variable('age2')
age3 = Variable('age3')
age4 = Variable('age4')
age5 = Variable('age5')
age6 = Variable('age6')
age7 = Variable('age7')
area = Variable('area')
avail1 = Variable('avail1')
avail2 = Variable('avail2')
avail3 = Variable('avail3')
avail4 = Variable('avail4')
avail5 = Variable('avail5')
choice = Variable('choice')
cost1 = Variable('cost1')
cost2 = Variable('cost2')
cost3 = Variable('cost3')
cost4 = Variable('cost4')
cost5 = Variable('cost5')
employ = Variable('employ')
inc = Variable('inc')
ones = Variable('ones')
status = Variable('status')
users = Variable('users')

low_income = database.DefineVariable('low_income', inc == 1)
medium_income = database.DefineVariable('medium_income', (inc >= 2) * (inc <= 4))
high_income = database.DefineVariable('high_income', inc == 5)
logcostBM = database.DefineVariable('logcostBM', log(cost1))
logcostSM = database.DefineVariable('logcostSM', log(cost2))
logcostLF = database.DefineVariable('logcostLF', log(cost3))
logcostEF = database.DefineVariable('logcostEF', log(cost4))
logcostMF = database.DefineVariable('logcostMF', log(cost5))
