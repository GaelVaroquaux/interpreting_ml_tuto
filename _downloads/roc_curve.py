"""
ROC curves on digit predictions
================================

"""
from sklearn import datasets, ensemble, metrics, model_selection, dummy
import matplotlib.pyplot as plt

digits = datasets.load_digits()

# First we work only on sevens:
sevens = (digits.target == 7)

classifier = ensemble.RandomForestClassifier()
most_frequent = dummy.DummyClassifier(strategy='most_frequent')

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    digits.data, sevens, random_state=0)

y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
most_frequent_score = most_frequent.fit(X_train,
                                        y_train).predict_proba(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test, y_score[:, 1])
roc_auc = metrics.auc(fpr, tpr)

fpr_dummy, tpr_dummy, _ = metrics.roc_curve(y_test,
                                            most_frequent_score[:, 1])
roc_auc_dummy = metrics.auc(fpr_dummy, tpr_dummy)

plt.figure(figsize=(3, 3))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='RandomForestClassifier\n(area = %0.2f)' % roc_auc)
plt.plot(fpr_dummy, tpr_dummy, color='.5',
         lw=lw, label='Dummy\n(area = %0.2f)' % roc_auc_dummy)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower left", title='ROC curve')
plt.tight_layout()
plt.show()

