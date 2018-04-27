"""
Interpreting linear models
==========================

Linear models are not that easy to interpret when variables are
correlated.

See also the statistics chapter of the scipy lecture notes

"""

#######################################################################
# Data on wages
# --------------
import urllib
import os
import pandas

if not os.path.exists('wages.txt'):
    # Download the file if it is not present
    urllib.urlretrieve('http://lib.stat.cmu.edu/datasets/CPS_85_Wages',
                       'wages.txt')

# Give names to the columns
names = [
    'EDUCATION: Number of years of education',
    'SOUTH: 1=Person lives in South, 0=Person lives elsewhere',
    'SEX: 1=Female, 0=Male',
    'EXPERIENCE: Number of years of work experience',
    'UNION: 1=Union member, 0=Not union member',
    'WAGE: Wage (dollars per hour)',
    'AGE: years',
    'RACE: 1=Other, 2=Hispanic, 3=White',
    'OCCUPATION: 1=Management, 2=Sales, 3=Clerical, 4=Service, 5=Professional, 6=Other',
    'SECTOR: 0=Other, 1=Manufacturing, 2=Construction',
    'MARR: 0=Unmarried,  1=Married',
]

short_names = [n.split(':')[0] for n in names]

data = pandas.read_csv('wages.txt', skiprows=27, skipfooter=6, sep=None,
                       header=None)
data.columns = short_names

# Log-transform the wages, as they typically increase with
# multiplicative factors
import numpy as np
data['WAGE'] = np.log10(data['WAGE'])

########################################################
# The challenge of correlated features
# --------------------------------------------
#
# Plot scatter matrices highlighting different aspects

import seaborn
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION', 'EXPERIENCE'])

########################################################################
# Note that age and experience are highly correlated
#
# A link between a single feature and the target is a *marginal* link.
#
#
# Univariate feature selection selects on marginal links.
#
# Linear model compute *conditional* links: removing the effects of other
# features on each feature. This is hard when features are correlated.


########################################################
# Coefficients of a linear model
# --------------------------------------------
#
from sklearn import linear_model
features = [c for c in data.columns if c != 'WAGE']
X = data[features]
y = data['WAGE']
ridge = linear_model.RidgeCV()
ridge.fit(X, y)

# Visualize the coefs
coefs = ridge.coef_
from matplotlib import pyplot as plt
plt.barh(np.arange(coefs.size), coefs)
plt.yticks(np.arange(coefs.size), features)
plt.tight_layout()

########################################################
# Note: coefs cannot easily be compared if X is not standardized: they
# should be normalized to the variance of X.
#
# When features are not too correlated and their is plenty, this is the
# well-known regime of standard statistics in linear models. Machine
# learning is not needs, and statsmodels is a great tool (see the
# statistics chapter in scipy-lectures)

########################################################
# The effect of regularization
# --------------------------------------------

lasso = linear_model.LassoCV()
lasso.fit(X, y)

coefs = lasso.coef_
from matplotlib import pyplot as plt
plt.barh(np.arange(coefs.size), coefs)
plt.yticks(np.arange(coefs.size), features)
plt.tight_layout()


