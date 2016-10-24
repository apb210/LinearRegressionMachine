import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import math as mth
from sklearn import linear_model
from statsmodels.stats.proportion import samplesize_confint_proportion

#For comparison with Least Square fit using Sklearn
reg = linear_model.LinearRegression()

#Read Data
data = pd.read_csv("housingdata.csv", header=None)

#Select what fraction of data needs to be used for training the model
train = data.sample(frac=0.7)

#The rest is selected for testing
test  = data.loc[~data.index.isin(train.index)]

#Rest the indices. Since the original indices are retained after cleaning the data
train.reset_index(drop=True, inplace = True)
test.reset_index(drop=True, inplace = True)


#Set the learning rate. 
alpha= 0.00001

#Since the 13th column is the price in the data, this is the Yvalues of the training set
trainY = train.iloc[:,13]

#The rest is Xdata
trainX = train.iloc[:,0:13]      #selects columns 0 through 12

#Similar treatment for the test data
testY = test.iloc[:,13]
testX = test.iloc[:,0:13]


#Using Least squares to fit the data from sklearn
reg.fit(trainX,trainY)
coeff = reg.coef_

yPredSkl = []
xindex = []
SQESkl = 0.0
#Taking the transpose for dot 
testX2 = testX.transpose()

#Testing how the Linear regression with Least Squares fared
for i in range(0, len(testX)):
    yPredSkl.append(testX2[i].dot(coeff))
    xindex.append(i)
    SQESkl = SQESkl + (yPredSkl[i]-testY[i])**2

#We'll plot later
RMSSkl=mth.sqrt(SQESkl/len(xindex))


# Now the data is in the format as shown below:

#===============================================================================
# Samples 0 1 2 3 4 5 ... m
# X1
# X2
# X3
# .
# .
# .
# XN
# 
# We would like to add a column for the intercept term in the LR 
#===============================================================================

 
# Setting the Zeroth column both for the training and the testing dataset.
# Prepare for the dummy variable for the intercept term
A0 = np.ones(trainX[0].count())  #Count the number of rows one has to input
A0t = np.ones(testX[0].count())

# The column indices get messed up. In the sense upon addition there is an extra 0th column
 
colindices = range(0,len(trainX.columns) + 1)
colindicest = range(0,len(testX.columns) + 1)

# Inserting the Zeroth Column
trainX.insert(loc = 0 ,column = 0 , value = A0, allow_duplicates=True)
testX.insert(loc = 0, column = 0, value = A0t, allow_duplicates=True)

# Resetting the column indices
trainX.columns = [colindices]
testX.columns = [colindicest]

# Taking the transpose for dot product 
trainY = trainY.transpose()
trainX = trainX.transpose()


testY = testY.transpose()
testX = testX.transpose()

# The coefficient vector
A = 0.1*np.random.rand(trainX[0].count())

Er = 10.0
count = 0

# This is the Stochastic Gradient Descent algorithm
#===============================================================================
# 
# i loops over random 
# j loops over the number of coefficients
# A = A + alpha * (ytrain(j-th sample) - trainX[i-th coefficeint][j - th sample])
#===============================================================================

print len(trainX)
while (Er > 0.5):
    count = count + 1
    
    j = randint(0, len(trainX.columns) - 1)

    for i in range(0, len(trainX)):
     
        Er = (trainY[j] - trainX[0].dot(A)) * trainX.iloc[i,j]
     
        A[i] = A[i] + (alpha / len(trainX)) * Er 
    
    # The error term
    Er = mth.sqrt((trainY[0] - trainX[0].dot(A))**2)
    if (count % 1000 == 0):
        print "Error =", Er



print "no of iterations: ", count

yPred = []
er = []
xdata = []
Err = []
SQE = 0.0
SQESkl = 0.0

# Checking our solution vector with the training set.  
for i in range(0, len(testX.columns)):
    yPred.append(testX[i].dot(A))
    xdata.append(i)
    SQE = SQE + (yPred[i] - testY[i])**2

    
RMS = mth.sqrt(SQE / len(xdata))

# Plotting
plt.plot(xdata,yPred,'ro',xdata,testY,'g^',xindex, yPredSkl,'bs')
plt.show()


