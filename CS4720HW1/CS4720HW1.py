# Morgan Knoch
# HW #1 CS4720

import scipy.io
import numpy as np
import math
from matplotlib import pyplot as plt
# import the data from .mat file
data = scipy.io.loadmat("data_class4.mat")

# get data array from dictionary
newdata = data['Data']

# get data for each class
firstClass = newdata[0][0]
secondClass = newdata[0][1]
thirdClass = newdata[0][2]
fourthClass = newdata[0][3]

#1 mean and covariance, for the second part use numpy.cov

# find mean
def findMean(data):
    mean = []
    for i in data:
        numinrow = 0
        totalrow = 0
        for j in i:
            numinrow += 1
            totalrow += j
        mean.append((totalrow / numinrow))
    return mean

def findCovariance(data, mean):
    dataFlipped = np.array(data)
    dataFlipped = dataFlipped.T

    meanarray = np.array(mean)
    
    numdata = 0

    covariance = np.zeros((2,2))
    
    for i in dataFlipped:
        newarray = np.array(i)
        
        ximinusmean = np.subtract(newarray, meanarray)
        ximinusmean = ximinusmean.reshape(2,1)
        ximinusmeantranspose = ximinusmean.T    

        covariance += np.matmul(ximinusmean, ximinusmeantranspose)

        numdata += 1

    return covariance/numdata 

# means of the classes
firstClassMean = findMean(firstClass)
secondClassMean = findMean(secondClass)
thirdClassMean = findMean(thirdClass)
fourthClassMean = findMean(fourthClass)

# covariances of the classes
firstClassCovariance = findCovariance(firstClass, firstClassMean)
firstClassCov = np.cov(firstClass)

secondClassCovariance = findCovariance(secondClass, secondClassMean)
secondClassCov = np.cov(secondClass)

thirdClassCovariance = findCovariance(thirdClass, thirdClassMean)
thirdClassCov = np.cov(thirdClass)

fourthClassCovariance = findCovariance(fourthClass, fourthClassMean)
fourthClassCov = np.cov(fourthClass)


#2 find the eigenvectors and eigenvalues for each class

# solve the equation Ev=yv, where y eigenvalues and x

def findEigenvectorsAndEigenValues(cov):
    # ax^2 + bx + c
    b = (-cov[0][0])-(cov[1][1])
    c = (cov[0][0] * cov[1][1]) - (cov[1][0] * cov[0][1])

    # produce the eigenvalues
    y1 = (-b + math.sqrt(b**2 - (4 * 1 * c)))/2
    y2 = (-b - math.sqrt(b**2 - (4 * 1 * c)))/2

    # Vectors
    v1 = []
    v2 = []

    # solve for first eigenvector
    if (cov[0][0]-y1) != 0 and cov[0][1] != 0:
        v1.append((cov[0][1]))
        v1.append((y1-cov[0][0]))
    else:
        v1.append((y1-cov[1][1]))
        v1.append((cov[1][0]))

    # solve for second eigenvector
    if (cov[0][0]-y2) != 0 and cov[0][1] != 0:
        v2.append((cov[0][1]))
        v2.append((y2-cov[0][0]))
    else:
        v2.append((y2-cov[1][1]))
        v2.append((cov[1][0]))

    # normalize vectors
    sumv1 = math.sqrt((v1[0]**2 + v1[1]**2))
    sumv2 = math.sqrt((v2[0]**2 + v2[1]**2))

    v1[0] = v1[0]/sumv1
    v1[1] = v1[1]/sumv1
    v2[0] = v2[0]/sumv2
    v2[1] = v2[1]/sumv2

    return v1, v2

firstClassV1, firstClassV2 = findEigenvectorsAndEigenValues(firstClassCovariance)
secondClassV1, secondClassV2 = findEigenvectorsAndEigenValues(secondClassCovariance)
thirdClassV1, thirdClassV2 = findEigenvectorsAndEigenValues(thirdClassCovariance)
fourthClassV1, fourthClassV2 = findEigenvectorsAndEigenValues(fourthClassCovariance)

# Graph the points and the vectors

# first class is blue
firstClass = np.array(firstClass)
firstClass = firstClass.T

for i in firstClass:
    plt.scatter(i[0],i[1], c='b')

# second class is red
secondClass = np.array(secondClass)
secondClass = secondClass.T

for i in secondClass:
    plt.scatter(i[0],i[1], c='r')

# third class is yellow
thirdClass = np.array(thirdClass)
thirdClass = thirdClass.T

for i in thirdClass:
    plt.scatter(i[0],i[1], c='y')

# fourth class is green
fourthClass = np.array(fourthClass)
fourthClass = fourthClass.T

for i in fourthClass:
    plt.scatter(i[0],i[1], c='g')

# plot eigenvectors
firstClassV1[0] = (firstClassV1[0]+firstClassMean[0])
firstClassV1[1] = (firstClassV1[1]+firstClassMean[1])
plt.quiver(firstClassMean[0], firstClassMean[1], firstClassV1[0], firstClassV1[1], angles='xy', scale_units='xy', scale=1)

firstClassV2[0] = (firstClassV2[0]+firstClassMean[0])
firstClassV2[1] = (firstClassV2[1]+firstClassMean[1])
plt.quiver(firstClassMean[0], firstClassMean[1], firstClassV2[0], firstClassV2[1], angles='xy', scale_units='xy', scale=1)


secondClassV1[0] = (secondClassV1[0]+secondClassMean[0])
secondClassV1[1] = (secondClassV1[1]+secondClassMean[1])
plt.quiver(secondClassMean[0], secondClassMean[1], secondClassV1[0], secondClassV1[1], angles='xy', scale_units='xy', scale=1)

secondClassV2[0] = (secondClassV2[0]+secondClassMean[0])
secondClassV2[1] = (secondClassV2[1]+secondClassMean[1])
plt.quiver(secondClassMean[0], secondClassMean[1], secondClassV2[0], secondClassV2[1], angles='xy', scale_units='xy', scale=1)


thirdClassV1[0] = (thirdClassV1[0]+thirdClassMean[0])
thirdClassV1[1] = (thirdClassV1[1]+thirdClassMean[1])
plt.quiver(thirdClassMean[0], thirdClassMean[1], thirdClassV1[0], thirdClassV1[1], angles='xy', scale_units='xy', scale=1)

thirdClassV2[0] = (thirdClassV2[0]+thirdClassMean[0])
thirdClassV2[1] = (thirdClassV2[1]+thirdClassMean[1])
plt.quiver(thirdClassMean[0], thirdClassMean[1], thirdClassV2[0], thirdClassV2[1], angles='xy', scale_units='xy', scale=1)

fourthClassV1[0] = (fourthClassV1[0]+fourthClassMean[0])
fourthClassV1[1] = (fourthClassV1[1]+fourthClassMean[1])
plt.quiver(fourthClassMean[0], fourthClassMean[1], fourthClassV1[0], fourthClassV1[1], angles='xy', scale_units='xy', scale=1)

fourthClassV2[0] = (fourthClassV2[0]+fourthClassMean[0])
fourthClassV2[1] = (fourthClassV2[1]+fourthClassMean[1])
plt.quiver(fourthClassMean[0], fourthClassMean[1], fourthClassV2[0], fourthClassV2[1], angles='xy', scale_units='xy', scale=1)

plt.xlim(-5, 20)
plt.ylim(-10, 10)

plt.show()

print('Done')