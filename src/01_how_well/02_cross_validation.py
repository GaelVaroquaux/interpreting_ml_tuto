"""
Cross-validation: some gotchas
===============================

Cross-validation is the ubiquitous test of a machine learning model. Yet
many things can go wrong.

"""

###############################################################
# The uncertainty of measured accuracy
# ------------------------------------
#
# The first thing to have in mind is that the results of a
# cross-validation are noisy estimate of the real prediction accuracy
#
# Let us create a simple artificial data
from sklearn import datasets, discriminant_analysis
import numpy as np
np.random.seed(0)
data, target = datasets.make_blobs(centers=[(0, 0), (0, 1)])
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
# This should not be surprising: if the classification rate is p, the
# observed distribution of correct classifications on a set of size
# follows a binomial distribution
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
#
# Importantly, the variance across folds is not a good measure of this
# error, as the different data folds are not independent. For instance,
# doing many random splits will can reduce the variance arbitrarily, but
# does not provide actually new data points

###############################################################
# Measuring baselines and chance
# -------------------------------
#
# Because of class imbalances, or confounding effects, it is easy to get
# it wrong it terms of what constitutes chances. There are two approaches
# to measure peformances of baselines or chance:
#
# Let's go back to simple generated data:
data, target = datasets.make_blobs(centers=[(0, 0), (0, 1)])

###############################################################
# **DummyClassifier** The dummy classifier:
# :class:`sklearn.dummy.DummyClassifier`, with different strategies to
# provide simple baselines
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="stratified")
print(cross_val_score(dummy, data, target))

###############################################################
# **Chance level** To measure actual chance, the most robust approach is
# to use permutations:
# :func:`sklearn.model_selection.permutation_test_score`, which is used
# as cross_val_score
from sklearn.model_selection import permutation_test_score
score, permuted_scores, p_value = permutation_test_score(classifier, data, target)
print("Classifier score: {0},\np value: {1}\nPermutation scores {2}"
        .format(score, p_value, permuted_scores))


###############################################################
# Non iid data and confounding effects
# -----------------------------------------
#
# Shool grading data
# ..................
#
# Download and load the data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Download some data on grades across several schools (centers)
#
# The junior school data, originally from http://www.bristol.ac.uk/cmm/learning/support/datasets/
import pandas as pd
import os
# Python 2 vs Python 3:
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

if not os.path.exists('exams.csv.gz'):
    # Download the file if it is not present
    urlretrieve('http://lib.stat.cmu.edu/datasets/CPS_85_Wages',
                'exams.csv.gz')
exams = pd.read_csv('exams.csv.gz')

# Select data for students present all three years
continuing_students = exams.StudentID.value_counts()
continuing_students = continuing_students[continuing_students > 2].index
exams = exams[exams.StudentID.isin(continuing_students)]

###############################################################
# Visualized factor of grades
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Grade at tests in in exams depend on socio-economic status, year at
# school, ...
import seaborn as sns
g = sns.PairGrid(exams.drop(columns=['StudentID']),
                 diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_upper(plt.scatter, s=2)
g.map_diag(sns.kdeplot, lw=3)

###############################################################
# A zoomed view on the factors that seem most interpretable
g = sns.PairGrid(exams[['Ravens', 'Maths', 'English', 'SocialClass', 'Year']],
                 diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_upper(plt.scatter, s=2)
g.map_diag(sns.kdeplot, lw=3)

###############################################################
# Predicting grades in maths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Can we predict test grades in maths from demographics (ie, not from
# other grades)?

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
# We get can predict!
#
# But there is one caveat: are we simply learning to recognive students
# across the years? There is many implicit informations about students:
# notably in the school ID and the class ID.
#
# To test for this, we can make sure that we have different students in
# the train and the test set
from sklearn import model_selection
cv = model_selection.GroupKFold(10)

print(cross_val_score(ensemble.GradientBoostingRegressor(), X, y,
                      cv=cv, groups=exams['StudentID']).mean())

###############################################################
# It works better!
#
# The classifier learns better to generalize, probably by learning
# stronger invariances from the repeated measures on the students
