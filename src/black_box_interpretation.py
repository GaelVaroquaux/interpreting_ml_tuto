"""
Black-box interpretation of models: LIME
=========================================

See also skater: a more modern variant relying on heavier dependencies
https://github.com/datascienceinc/Skater

First we need to install lime with the following shell line::

    $ pip install --user lime

Lime uses the notion of "explainers", for different types of data,
tabular, image, text.

"""

##########################################################
# Regression on tabular data: factors of prices of houses
# --------------------------------------------------------
#
# Load the data, create and fit a regressor
from sklearn import datasets, ensemble, model_selection

boston = datasets.load_boston()
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    boston.data, boston.target)

regressor = ensemble.RandomForestRegressor()
regressor.fit(X_train, y_train)

##########################################################
# Inspect predictions for a few houses
#
# For this, separate out the categorical features:
import numpy as np
categorical_features = [i for i, col in enumerate(boston.data.T)
                        if np.unique(col).size < 10]

##########################################################
# Now use a lime explainer for tabular data
from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(X_train,
    feature_names=boston.feature_names,
    class_names=['price'],
    categorical_features=categorical_features,
    mode='regression')

# Now explain a prediction
exp = explainer.explain_instance(X_test[25], regressor.predict,
        num_features=10)

exp.as_pyplot_figure()
from matplotlib import pyplot as plt
plt.tight_layout()
##########################################################
print(exp.as_list())

##########################################################
# Explain a few more predictions
for i in [7, 50, 66]:
    exp = explainer.explain_instance(X_test[i], regressor.predict,
            num_features=10)
    exp.as_pyplot_figure()
    plt.tight_layout()



