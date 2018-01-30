# Morgan Knoch
# HW #1 CS4720

import scipy.io
import numpy as np

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






print('Done')