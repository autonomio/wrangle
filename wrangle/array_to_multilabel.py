import numpy as np


def array_to_multilabel(y):

	'''Converts a 1-d array with multiclass
	data into corresponding 2-d multilabel
	version.

	'''
	size = len(y)
	b = np.zeros((size, max(y) + 1))
	b[np.arange(size), y] = 1

	return b