__author__ = 'prabhath'

import pandas as pd


class MainCollection(object):

    target_url = ("""http://archive.ics.uci.edu/ml/machine-""" \
"""learning-databases/wine-quality/winequality-red.csv""")

    def get_data(self):
        return pd.read_csv(MainCollection.target_url, header=0, sep=";")
