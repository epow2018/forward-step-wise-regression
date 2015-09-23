__author__ = 'prabhath'

from main import MainCollection
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


"""
Forward step wise regression i.e modified OLS(Ordinary least squares)
use best subset selection to select subset of features by best subset
selection from the predictors based on the to avoid overfitting( there
is a signinficant discrepency between errors on training data and errors on test data.)
"""

# 1. Getting the data.
df = MainCollection()
data = df.get_data()
xLists = []
labels = []
names = pd.Series(data.columns) # to get all the values names.values will do the trick.s
firstline = True

for line in data.values:
    row = list(line)
    # Add labels to labels list
    labels.append(row[-1])
    # Remove label
    row.pop()
    # Convert into float
    floatrow = [float(s) for s in row]
    xLists.append(floatrow)

# 2. Divide data into training and test daya set.
indices = range(len(xLists))

xListtrain = [xLists[i] for i in indices if i % 3 != 0]
xListtest = [xLists[i] for i in indices if i % 3 == 0]
labeltest = [labels[i] for i in indices if i % 3 == 0]
labeltrain = [labels[i] for i in indices if i % 3 != 0]

# 3.
""" Build list of attributes on-at-a-time starting with empty. """