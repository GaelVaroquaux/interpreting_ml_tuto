"""
Cross-validation: some gotchas
===============================

Cross-validation is the ubiquitous test of a machine learning model. Yet
many things can go wrong.

.. contents::
    :local:

"""

###############################################################
# Uncertainty of measured accuracy
# --------------------------------
#
# Variations in cross_val_score: simple experiments
# ..................................................
#
# The first thing to have in mind is that the results of a
# cross-validation are noisy estimate of the real prediction accuracy
#
# Let us create a simple artificial data
from sklearn import datasets, discriminant_analysis
import numpy as np
np.random.seed(0)
data, target = datasets.make_blobs(centers=[(0, 0), (0, 1)], n_samples=100)
classifier = discriminant_analysis.LinearDiscriminantAnalysis()

###############################################################
# One cross-validation gives spread out measures
from sklearn.model_selection import cross_val_score
print(cross_val_score(classifier, data, target))

###############################################################
# What if we try different random shuffles of the data?
from sklearn import utils
for _ in range(10):
    data, target = utils.shuffle(data, target)
    print(cross_val_score(classifier, data, target))

###############################################################
# A simple probabilistic model
# .............................
#
# A sample probabilistic model gives the distribution of observed error:
# if the classification rate is p, the observed distribution of correct
# classifications on a set of size follows a binomial distribution
from scipy import stats
n = len(data)
distrib = stats.binom(n=n, p=.7)

###############################################################
# We can plot it:
from matplotlib import pyplot as plt
plt.figure(figsize=(6, 3))
plt.plot(np.linspace(0, 1, n), distrib.pmf(np.arange(0, n)))

###############################################################
# It is wide, because there are not that many samples to mesure the error
# upon: this is a small dataset.
#
# We can look at the interval in which 95% of the observed accuracy lies
# for different sample sizes
for n in [100, 1000, 10000, 100000, 1000000]:
    distrib = stats.binom(n, .7)
    interval = (distrib.isf(.025) - distrib.isf(.975)) / n
    print("Size: {0: 8}  | interval: {1}%".format(n, 100 * interval))

###############################################################
# At 100 000 samples, 5% of the observed classification accuracy still
# fall more than .5% away of the true rate.
#
# **Keep in mind that cross-val is a noisy measure**

###############################################################
# Empirical distribution of cross-validation scores
# .....................................................
#
# We can sample the distribution of scores using cross-validation
# iterators based on subsampling, such as
# :class:`sklearn.model_selection.ShuffleSplit`, with many many splits
from sklearn import model_selection
cv = model_selection.ShuffleSplit(n_splits=200)
scores = cross_val_score(classifier, data, target, cv=cv)

import seaborn as sns
plt.figure(figsize=(6, 3))
sns.distplot(scores)
plt.xlim(0, 1)

###############################################################
# The empirical distribution is broader than the theoretical one. This
# can be explained by the fact that as we are retraining the model on
# each fold, it actually fluctuates due the sampling noise in the
# training data, while the model above only accounts for sampling noise
# in the test data.
#
# The situation does get better with more data:
data, target = datasets.make_blobs(centers=[(0, 0), (0, 1)], n_samples=1000)

scores = cross_val_score(classifier, data, target, cv=cv)

plt.figure(figsize=(6, 3))
sns.distplot(scores)
plt.xlim(0, 1)
plt.title("Distribution with 1000 data points")

###############################################################
# The distribution is still very broader
#
# **Testing the observed scores**
#
# Importantly, the standard error of the mean across folds is not a good
# measure of this error, as the different data folds are not independent.
# For instance, doing many random splits will can reduce the variance
# arbitrarily, but does not provide actually new data points
from scipy import stats

plt.figure(figsize=(6, 3))
sns.distplot(scores)
plt.axvline(np.mean(scores), color='k')
plt.axvline(np.mean(scores) + np.std(scores), color='b', label='std')
plt.axvline(np.mean(scores) - np.std(scores), color='b')
plt.axvline(np.mean(scores) + stats.sem(scores), color='r', label='SEM')
plt.axvline(np.mean(scores) - stats.sem(scores), color='r')
plt.legend(loc='best')
plt.xlim(0, 1)
plt.title("Distribution with 1000 data points")

###############################################################
# Measuring baselines and chance
# ...............................
#
# Because of class imbalances, or confounding effects, it is easy to get
# it wrong it terms of what constitutes chances. There are two approaches
# to measure peformances of baselines or chance:

###############################################################
# **DummyClassifier** The dummy classifier:
# :class:`sklearn.dummy.DummyClassifier`, with different strategies to
# provide simple baselines
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="stratified")
dummy_scores = cross_val_score(dummy, data, target)
print(dummy_scores)

###############################################################
# **Chance level** To measure actual chance, the most robust approach is
# to use permutations:
# :func:`sklearn.model_selection.permutation_test_score`, which is used
# as cross_val_score
from sklearn.model_selection import permutation_test_score
score, permuted_scores, p_value = permutation_test_score(classifier,
                                                         data, target)
print("Classifier score: {0},\np value: {1}\nPermutation scores {2}"
        .format(score, p_value, permuted_scores))

###############################################################
# We can plot all the scores
plt.figure(figsize=(6, 3))
sns.distplot(dummy_scores, color="g", label="Dummy scores")
sns.distplot(permuted_scores, color="r", label="Permuted scores")
sns.distplot(scores, label="Cross validation scores")
plt.legend(loc='best')
plt.xlim(0, 1)

###############################################################
# Permutation and performing many cross-validation splits are
# computationally expensive, but they give trust-worthy answers

###############################################################
# Cross-validation with non iid data
# -----------------------------------
#
# Another common caveat for cross-validation are dependencies in the
# observations that can easily creep in between the train and the test
# sets. Let us explore these problems in two settings.
#
# Stock market: time series
# ...........................
#
# **Download**: Download and load the data:
import pandas as pd
import os
# Python 2 vs Python 3:
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

symbols = {'TOT': 'Total', 'XOM': 'Exxon', 'CVX': 'Chevron',
           'COP': 'ConocoPhillips', 'VLO': 'Valero Energy'}

quotes = pd.DataFrame()

for symbol, name in symbols.items():
    url = ('https://raw.githubusercontent.com/scikit-learn/examples-data/'
           'master/financial-data/{}.csv')
    filename = "{}.csv".format(symbol)
    if not os.path.exists(filename):
        urlretrieve(url.format(symbol), filename)
    this_quote = pd.read_csv(filename)
    quotes[name] = this_quote['open']

###############################################################
# **Prediction**: Predict 'Chevron' from the others
from sklearn import linear_model, model_selection, ensemble
cv = model_selection.ShuffleSplit(random_state=0)
print(cross_val_score(linear_model.RidgeCV(),
                      quotes.drop(columns=['Chevron']),
                      quotes['Chevron'],
                      cv=cv).mean())

###############################################################
# Is this a robust prediction?
#
# Does it cary over across quarters?
#
# **Stratification**: To thest this we need to stratify cross-validation
# using a :class:`sklearn.model_selection.LeaveOneGroupOut`
quarters = pd.to_datetime(this_quote['date']).dt.to_period('Q')
cv = model_selection.LeaveOneGroupOut()

print(cross_val_score(linear_model.RidgeCV(),
                      quotes.drop(columns=['Chevron']),
                      quotes['Chevron'],
                      cv=cv, groups=quarters).mean())

###############################################################
# The problem that we are facing here is the auto-correlation in the
# data: these datasets are **time-series**.
quotes_with_dates = pd.concat((quotes, this_quote['date']),
                              axis=1).set_index('date')
quotes_with_dates.plot()

###############################################################
# **Testing for forecasting**: If the goal is to do forecasting, than
# prediction should be done in the future, for instance using
# :class:`sklearn.model_selection.TimeSeriesSplit`
#
# Can we do forecasting: predict the future?

cv = model_selection.TimeSeriesSplit(n_splits=quarters.nunique())

print(cross_val_score(linear_model.RidgeCV(),
                      quotes.drop(columns=['Chevron']),
                      quotes['Chevron'],
                      cv=cv, groups=quarters).mean())

###############################################################
# No. This prediction is abysmal.

###############################################################
# School grades: repeated measures
# .................................
#
# Let us look at another dependency structure across samples: repeated
# measures. This is often often in longitudinal data. Here we are looking
# at grades of school students, across the years.
#
# **Download** First we download some data on grades across several
# schools (centers)
#
# The junior school data, originally from http://www.bristol.ac.uk/cmm/learning/support/datasets/
if not os.path.exists('exams.csv.gz'):
    # Download the file if it is not present
    urlretrieve('https://raw.githubusercontent.com/GaelVaroquaux/interpreting_ml_tuto/blob/master/src/01_how_well/exams.csv.gz',
                filename)
exams = pd.read_csv('exams.csv.gz')

# Select data for students present all three years
continuing_students = exams.StudentID.value_counts()
continuing_students = continuing_students[continuing_students > 2].index
exams = exams[exams.StudentID.isin(continuing_students)]

###############################################################
# **Visualization**: Grade at tests in in exams depend on socio-economic
# status, year at school, ...
#
# The simplest way to do this is using seaborn's pairplot function.

import seaborn as sns
sns.pairplot(exams.drop(columns=['StudentID']))

###############################################################
# A more elaborate plot using density estimation gives better
# understanding of the dense regions:
g = sns.PairGrid(exams.drop(columns=['StudentID']),
                 diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_upper(plt.scatter, s=2)
g.map_diag(sns.kdeplot, lw=3)


###############################################################
# **Prediction**: Can we predict test grades in maths from demographics
# (ie, not from other grades)?

# A bit of feature engineering to get a numerical matrix (easily done
# with the ColumnTransformer in scikit-learn >= 0.20)
X = exams.drop(columns=['StudentID', 'Maths', 'Ravens', 'English'])
# Encode gender as an integer variables
X['Gender'] = X['Gender'] == 'Girl'
# One-hot encode social class
X = pd.get_dummies(X, drop_first=True)
y = exams['Maths']

from sklearn import ensemble
print(cross_val_score(ensemble.GradientBoostingRegressor(), X, y,
                      cv=10).mean())

###############################################################
# We can predict!
#
# But there is one caveat: are we simply learning to recognive students
# across the years? There is many implicit informations about students:
# notably in the school ID and the class ID.
#
# **Stratification** To test for this, we can make sure that we have
# different students in the train and the test set.
from sklearn import model_selection
cv = model_selection.GroupKFold(10)

print(cross_val_score(ensemble.GradientBoostingRegressor(), X, y,
                      cv=cv, groups=exams['StudentID']).mean())

###############################################################
# It works better!
#
# The classifier learns better to generalize, probably by learning
# stronger invariances from the repeated measures on the students
#
# Summary
# .......
#
# Samples often have a dependency structure, such a with time-series, or
# with repeated measures. To have a meaningful measure of prediction
# error, the link between the train and the test set must match the
# important one for the application. In time-series prediction, it must
# be in the future. To learn a predictor of the success of an individual
# from demographics, it might be more relevant to predict across
# individuals. If the variance across individuals is much larger than the
# variance across repeated measurement, as in many biomedical
# applications, the choice of cross-validation strategy may make a huge
# difference.

