# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 00:45:44 2015

@author: Kruthika
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

loansData = pd.read_csv('LoanStats3d.csv', skiprows=1, skip_footer=2)

cleanInterestRate = loansData['int_rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
loansData['int_rate'] = cleanInterestRate

intrate = loansData['int_rate']
income = loansData['annual_inc']
homeOwnership = loansData['home_ownership']

#To model interest rate with annual income
y = np.matrix(intrate).transpose()
model = sm.OLS(y,income)
f = model.fit()
f.summary()

#To model interest rate with annual income and home ownership
#x1 = np.matrix(income).transpose()
#x2 = np.matrix(homeOwnership).transpose()

#x = np.column_stack([x1,x2])
#X = sm.add_constant(x)
model = sm.ols(formula = "int_rate ~ annual_inc + home_ownership", data = loansData)
f1 = model.fit()
     
f1.summary()
