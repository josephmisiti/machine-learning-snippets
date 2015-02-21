
from sklearn import datasets


def get_digits_dataset():
	""" gets digits data set"""
	digits = datasets.load_digits()
	X,y =  digits.data, digits.target
	return X,y