"""
Metrics to judge the sucess of a model
=======================================

Pro & cons of various performance metrics.

.. contents::

The simple way to use a scoring metric during cross-validation is
via the `scoring` parameter of
:func:`sklearn.model_selection.cross_val_score`.
"""


#############################################################
# Regression settings
# -----------------------
#
# The Boston housing data
# ........................

from sklearn import datasets
boston = datasets.load_boston()

# Shuffle the data
from sklearn.utils import shuffle
data, target = shuffle(boston.data, boston.target, random_state=0)

#############################################################
# A quick plot of how each feature is related to the target
from matplotlib import pyplot as plt

for feature, name in zip(data.T, boston.feature_names):
    plt.figure(figsize=(4, 3))
    plt.scatter(feature, target)
    plt.xlabel(name, size=22)
    plt.ylabel('Price (US$)', size=22)
    plt.tight_layout()

#############################################################
# We will be using a random forest regressor to predict the price
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()

#############################################################
# Explained variance vs Mean Square Error
# .......................................
#
# The default score is explained variance
from sklearn.model_selection import cross_val_score
print(cross_val_score(regressor, data, target))

#############################################################
# Explained variance is convienent because it has a natural scaling: 1 is
# perfect prediction, and 0 is around chance
#
# Now let us see which houses are easier to predict:
#
# Not along the Charles river (feature 3)
print(cross_val_score(regressor, data[data[:, 3] == 0],
                      target[data[:, 3] == 0]))

#############################################################
# Along the Charles river
print(cross_val_score(regressor, data[data[:, 3] == 1],
                      target[data[:, 3] == 1]))

#############################################################
# So the houses along the Charles are harder to predict?
#
# It's not so easy to conclude this from the explained variance: in two
# different sets of observations, the variance of the target differs, and
# the explained variance is a relative measure
#
# **MSE**: We can use the mean squared error (here negated)
#
# Not along the Charles river
print(cross_val_score(regressor, data[data[:, 3] == 0],
                      target[data[:, 3] == 0],
                      scoring='neg_mean_squared_error'))

#############################################################
# Along the Charles river
print(cross_val_score(regressor, data[data[:, 3] == 1],
                      target[data[:, 3] == 1],
                      scoring='neg_mean_squared_error'))

#############################################################
# So the error is larger along the Charles river


#############################################################
# Mean Squared Error versus Mean Absolute Error
# ..................................................
#
# What if we want to report an error in dollars, meaningful for an
# application?
#
# The Mean Absolute Error is useful for this goal
print(cross_val_score(regressor, data, target,
                      scoring='neg_mean_absolute_error'))

#############################################################
# Summary
# .........
#
# * **explained variance**: scaled with regards to chance: 1 = perfect,
#   0 = around chance, but it shouldn't used to compare predictions
#   across datasets
#
# * **mean absolute error**: enables comparison across datasets in the
#   units of the target

#############################################################
# Classification settings
# -----------------------
#
# The digits data
# .................
digits = datasets.load_digits()
# Let us try to detect sevens:
sevens = (digits.target == 7)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

#############################################################
# Accuracy and its shortcomings
# .............................
#
# The default metric is the accuracy: the averaged fraction of success.
# It takes values between 0 and 1, where 1 is perfect prediction
print(cross_val_score(classifier, digits.data, sevens))

#############################################################
# However, a stupid classifier can each good prediction wit imbalanced
# classes
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent')
print(cross_val_score(dummy, digits.data, sevens))

#############################################################
# Balanced accuracy (available in development scikit-learn versions)
# fixes this, but can have surprising behaviors, such as being negative

#############################################################
# Precision, recall, and their shortcomings
# ..........................................
#
# We can measure separately false detection and misses
#
# **Precision**: Precision counts the ratio of detections that are
# correct
print(cross_val_score(classifier, digits.data, sevens,
                      scoring='precision'))
#############################################################
# Our classifier has a good precision: most of the sevens that it
# predicts are really sevens.
#
# As predicting the most frequent never predicts sevens, precision is ill
# defined. Scikit-learn puts it to zero
print(cross_val_score(dummy, digits.data, sevens, scoring='precision'))


#############################################################
# **Recall**: Recall counts the fraction of class 1 actually detected
print(cross_val_score(classifier, digits.data, sevens, scoring='recall'))

#############################################################
# Our recall isn't as good: we miss many sevens
#
# But predicting the most frequent never predicts sevens:
print(cross_val_score(dummy, digits.data, sevens, scoring='recall'))

#############################################################
#
# **Note** Measuring only the precision without the recall makes no
# sense, it is easy to maximize one at the cost of the other. Ideally,
# classifiers should be compared on a precision at a given recall

#############################################################
# Area under the ROC curve
# ..........................
#
# If the classifier provides a decision function that can be thresholded
# to control false positives versus false negatives, the ROC curve
# summarise the different tradeoffs that can be achieved by varying this
# threshold.
#
# Its Area Under the Curve (AUC) is a useful metric where 1 is perfect
# prediction and .5 is chance, independently of class imbalance
print(cross_val_score(classifier, digits.data, sevens, scoring='roc_auc'))


#############################################################
# Average precision
# ..................

#############################################################
# Multiclass and multilabel settings
# ...................................
