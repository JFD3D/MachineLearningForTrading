#Code for testing KNNLearner
#Author: Arvind Sundararajan
#
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import math
from KNNLearner import *


#While searching for the error relating to total_seconds() in python 2.6, 
#i came across the following link 
#http://stackoverflow.com/questions/3694835/python-2-6-5-divide-timedelta-with-timedelta, 
#but i wrote this trivial function from scratch and 
#tested its behaviour on python 2.7 with total_seconds() function
def gettotalseconds(starttime, endtime):
  delta = endtime - starttime
  deltatotalseconds = (delta.microseconds + (delta.seconds + delta.days*24*3600)*1e6)/1e6

  return deltatotalseconds

def main():
  trainpercent = 60
  isRandomSplit = False

  filenames = ['data-classification-prob.csv', 'data-ripple-prob.csv']
  outputfilenames = ['plot1.pdf', 'plot2.pdf']
  trainfilenames = ['traintime1.pdf', 'traintime2.pdf']
  testfilenames = ['testtime1.pdf', 'testtime2.pdf']
  methods = ['mean','median']
  
  for index in range(2):
    #read data from data file
    input = np.loadtxt(filenames[index], delimiter=',')
    trainsize = math.floor(input.shape[0]*trainpercent/100)
  
    #split data into train and test sets 
    Xtrain = input[0:trainsize,:-1]
    Ytrain = input[0:trainsize,-1]
    Xtest = input[trainsize:,:-1]
    Ytest = input[trainsize:,-1]
  
    MAXK = 300
    NUMCOLS = 4
    
    meanstats = np.zeros((MAXK, NUMCOLS), dtype=np.float)
    medianstats = np.zeros((MAXK, NUMCOLS), dtype=np.float)
    
    avgtraintime = -1
    avgtesttime = -1
  
    for method in methods:
      stats = np.zeros((MAXK, NUMCOLS), dtype=np.float)
      bestcorr = -1000
      bestK = -1
      
      for k in range(1, MAXK+1):
        #instantiate learner and test
        learner = KNNLearner(k, method)
      
        #get start time
        trainstarttime = dt.datetime.now()
        learner.addEvidence(Xtrain, Ytrain)
        #get end time and print total time for adding evidnece
        trainendtime = dt.datetime.now()
      
        #get start time
        teststarttime = dt.datetime.now()
        Y = learner.query(Xtest)
        #get end time and print total time for testing
        testendtime = dt.datetime.now()
    
        #compute corrcoef
        corr = np.corrcoef(Ytest.T, Y.T)
        if corr[0,1] > bestcorr:
          bestcorr = corr[0,1]
          bestK = k
      
        stats[k-1, 0] = k
        stats[k-1, 1] = corr[0,1]
        #The total_seconds() method works in python >= 2.7
        #stats[k-1, 2] = (trainendtime - trainstarttime).total_seconds()/Xtrain.shape[0]
        #stats[k-1, 3] = (testendtime - teststarttime).total_seconds()/Xtest.shape[0]
        stats[k-1, 2] = gettotalseconds(trainstarttime, trainendtime)/Xtrain.shape[0]
        stats[k-1, 3] = gettotalseconds(teststarttime, testendtime)/Xtest.shape[0]
      
        if k == 3 and method == 'mean':
          avgtraintime = stats[k-1,2]
          avgtesttime = stats[k-1,3]

      print 'File:%s Method:%s BestCorrelation:%f K corresponding to best correlation:%f AvgTrainTimeForK3Mean :%f seconds AvgTestTimeForK3Mean:%f seconds'%(filenames[index], method, bestcorr, bestK, avgtraintime, avgtesttime)
    
      if method == 'median':
        medianstats = stats.copy()
      else: 
        meanstats = stats.copy()
    
    timedelta = 1

    #Graph for k versus corrcoef
    plt.cla()
    plt.clf()
    plt.plot(meanstats[:,0], meanstats[:,1], color='r')
    plt.plot(medianstats[:,0], medianstats[:,1], color='b')
    plt.legend(('method=mean', 'method=median'), loc='upper right')
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('k')
    plt.savefig(outputfilenames[index],format='pdf')

    #Graph for k versus time for training
    #plt.cla()
    #plt.clf()
    #plt.plot(stats[:,0], stats[:,2], color = 'r')
    #plt.ylabel('Time taken for training in seconds')
    #plt.xlabel('k')
    #plt.ylim(min(stats[:,2])-timedelta, max(stats[:,2])+timedelta)
    #plt.savefig(trainfilenames[index],format='pdf')
  
    #Graph for k versus time for testing
    #plt.cla()
    #plt.clf()
    #plt.plot(stats[:,0], stats[:,3], color='b')
    #plt.ylabel('Time taken for testing in seconds')
    #plt.xlabel('k')
    #plt.ylim(min(stats[:,3])-timedelta, max(stats[:,3])+timedelta)
    #plt.savefig(testfilenames[index],format='pdf')
  

if __name__=='__main__':
  main()
