import pprint
from utils import *
from sklearn import cross_validation
from sklearn import svm
from sklearn import linear_model
import numpy as np

X,y = get_digits_dataset()
k_fold = cross_validation.KFold(n=6, n_folds=3)
for train_indices, test_indices in k_fold:
	print "training={} testing={}".format(train_indices,test_indices)
	
	
kfold = cross_validation.KFold(n=len(y), n_folds=3)

svc = svm.SVC(C=1, kernel='linear')
logrc = linear_model.LogisticRegression()

for train, test in kfold:	
	print svc.fit(X[train], y[train]).score(X[test], y[test])
	print logrc.fit(X[train], y[train]).score(X[test], y[test])
	
	
##################
# Example 1 (http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html)
#################

svc = svm.SVC(kernel='linear')
Cs = np.logspace(-10, 0, 10)

kfold = cross_validation.KFold(n=len(y), n_folds=3)
results = {}
for C in Cs:
	print "training SVM w/ cost={}".format(C)
	for train, test in kfold:	
		results.setdefault(str(C),[])
		svc = svm.SVC(C=C, kernel='linear')
		results[str(C)].append(svc.fit(X[train], y[train]).score(X[test], y[test]))

for key in results.keys():
	print "cost={} mean score={}".format(key, np.mean(results[key]))
	
		

