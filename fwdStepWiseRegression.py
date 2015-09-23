__author__ = 'prabhath'

from main import MainCollection
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import numpy as np

"""
Forward step wise regression i.e modified OLS(Ordinary least squares)
use best subset selection to select subset of features by best subset
selection from the predictors based on the to avoid overfitting( there
is a signinficant discrepency between errors on training data and errors on test data.)
"""

# 1. Getting the data.
df = MainCollection()
data = df.get_data()

