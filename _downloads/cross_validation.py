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
# This should not be surprising: if the lassification rate is p, the
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
# upon: iris is a small dataset
#
# We can look at the interval in which 95% of the observed accuracy lies
# for different sample sizes
for n in [100, 1000, 10000, 100000]:
    distrib = stats.binom(n, .7)
    interval = (distrib.isf(.025) - distrib.isf(.975)) / n
    print("Size: {0: 7}  | interval: {1}%".format(n, 100 * interval))

###############################################################
# At 100 000 samples, 5% of the observed classification accuracy still
# fall more than .5% away of the true rate
#
# **Keep in mind that cross-val is a noisy measure**
#
# Importantly, the variance across folds is not a good measure of this
# error, as the different data folds are not independent. For instance,
# doing many random splits will can reduce the variance arbitrarily, but
# does not provide actually new data points

###############################################################
# Confounding effects and non independence
# -----------------------------------------

###############################################################
# Measuring baselines and chance
# -------------------------------
#
# Because of class imbalances, or confounding effects, it is easy to get
# it wrong it terms of what constitutes chances. There are two approaches
# to measure peformances of baselines or chance:
#
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

