__author__ = 'prabhath'

from main import MainCollection
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


def xattrSelect(x, idxset):
    """ Takes X matrix as list of list and returns subset containing columns in idxSet """
    xout = []
    for row in x:
        xout.append([row[i] for i in idxset])
    return xout


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
names = pd.Series(data.columns) # to get all the values names.values will do the trick.
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
attributeList = []
index = range(len(xLists[1])) # gives number of columns
indexSet = set(index)
indexSeq = []
oosError = []

for i in index:
    attSet = set(attributeList)
    # attributes not in list.
    attTryset = indexSet - attSet
    # form into list
    attTry = [o for o in attTryset]
    errorList = []
    attTemp = []
    # try for each one not in set to
    # see with will giveleast oos error
    for j in attTry:
        attTemp = [] + attributeList
        attTemp.append(j)

        # Use the attTemp to form training and testing sub matrixes lists of lists
        xTraintemp = xattrSelect(xListtrain, attTemp)
        xTesttemp = xattrSelect(xListtest, attTemp)

        # Convert into arrays
        xTrain = np.array(xTraintemp)
        yTrain = np.array(labeltrain)
        xTest = np.array(xTesttemp)
        yTest = np.array(labeltest)

        # use scikit learn linear regression
        wineQmodel = linear_model.LinearRegression()
        wineQmodel.fit(xTrain, yTrain)

        # Use trained model to generate prediction and calculate rmsError
        rmsError = np.linalg.norm((yTest - wineQmodel.predict(xTest)), 2) / math.sqrt(len(yTest))
        errorList.append(rmsError)
        attTemp = []

    iBest = np.argmin(errorList)
    attributeList.append(attTry[iBest])
    oosError.append(errorList[iBest])

print("Out of sample error versus attribute set size" )
print(oosError)
print("\n" + "Best attribute indices")
print(attributeList)
namesList = [names[i] for i in attributeList]
print("\n" + "Best attribute names")
print(namesList)

# Plot error vs number of attributes.
x = range(len(oosError))
plt.plot(x, oosError, 'k')
plt.xlabel('Number of Attributes')
plt.ylabel('Error (RMS)')
plt.show()

#Plot histogram of out of sample errors for best number of attributes
#Identify index corresponding to min value,
#retrain with the corresponding attributes
#Use resulting model to predict against out of sample data.
#Plot errors (aka residuals)
indexBest = oosError.index(min(oosError))
attributesBest = attributeList[1:(indexBest+1)]
#Define column-wise subsets of xListTrain and xListTest
#and convert to numpy
xTrainTemp = xattrSelect(xListtrain, attributesBest)
xTestTemp = xattrSelect(xListtest, attributesBest)
xTrain = np.array(xTrainTemp); xTest = np.array(xTestTemp)
#train and plot error histogram
wineQModel = linear_model.LinearRegression()
wineQModel.fit(xTrain,yTrain)
errorVector = yTest-wineQModel.predict(xTest)
plt.hist(errorVector)
plt.xlabel("Bin Boundaries")
plt.ylabel("Counts")
plt.show()

#scatter plot of actual versus predicted
plt.scatter(wineQModel.predict(xTest), yTest, s=100, alpha=0.10)
plt.xlabel('Predicted Taste Score')
plt.ylabel('Actual Taste Score')
plt.show()

