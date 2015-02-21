import pprint
from utils import *
from sklearn import cross_validation
from sklearn import svm
from sklearn import linear_model
import numpy as np

X,y = get_digits_dataset()

cv = cross_validation.StratifiedKFold(y, 2)

# define an svm and a logistic regression
svc = svm.SVC(C=1, kernel='linear')
logrc = linear_model.LogisticRegression()

for train, test in cv:
	print svc.fit(X[train], y[train]).score(X[test], y[test])
	print logrc.fit(X[train], y[train]).score(X[test], y[test])
	