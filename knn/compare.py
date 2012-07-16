#Code for comparing KNNLearner vs KDTKNNLearner
#Author: Arvind Sundararajan
#
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import math
from KNNLearner import *
from qstklearn.kdtknn import *

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
  methods = ['mean','median']
  
  #read data from data file
  input = np.loadtxt('data-ripple-prob.csv', delimiter=',')
  trainsize = math.floor(input.shape[0]*trainpercent/100)

  #split data into train and test sets 
  Xtrain = input[0:trainsize,:-1]
  Ytrain = input[0:trainsize,-1]
  Xtest = input[trainsize:,:-1]
  Ytest = input[trainsize:,-1]

  MAXK = 30
  NUMCOLS = 5
  
  meanstats = np.zeros((MAXK, NUMCOLS), dtype=np.float)
  medianstats = np.zeros((MAXK, NUMCOLS), dtype=np.float)
  
  for method in methods:
    stats = np.zeros((MAXK, NUMCOLS), dtype=np.float)
    
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
      
      stats[k-1, 0] = k
      stats[k-1, 1] = gettotalseconds(trainstarttime, trainendtime)/Xtrain.shape[0]
      stats[k-1, 2] = gettotalseconds(teststarttime, testendtime)/Xtest.shape[0]
      
      kdtlearner = kdtknn(k, method)
      #get start time
      trainstarttime = dt.datetime.now()
      kdtlearner.addEvidence(Xtrain, Ytrain)
      #get end time and print total time for adding evidnece
      trainendtime = dt.datetime.now()
    
      #get start time
      teststarttime = dt.datetime.now()
      Y = kdtlearner.query(Xtest)
      #get end time and print total time for testing
      testendtime = dt.datetime.now()
      
      stats[k-1, 3] = gettotalseconds(trainstarttime, trainendtime)/Xtrain.shape[0]
      stats[k-1, 4] = gettotalseconds(teststarttime, testendtime)/Xtest.shape[0]
      
    if method == 'median':
      medianstats = stats.copy()
    else: 
      meanstats = stats.copy()
  
  #Graph for time/instance versus corrcoef
  timedelta = 0.001
  outputfilenames = ['mytraining.pdf', 'myquery.pdf', 'kdtknntraining.pdf', 'kdtknnquery.pdf']
  titles = ['mytrainingtime/instance', 'myquerytime/instance', 'kdtknntrainingtime/instance', 'kdtknnquerytime/instance']
  for index in range(1, NUMCOLS):
    plt.cla()
    plt.clf()
    plt.plot(meanstats[:,0], meanstats[:,index], color='r')
    plt.plot(medianstats[:,0], medianstats[:,index], color='b')
    plt.legend(('method=mean', 'method=median'), loc='upper right')
    plt.ylabel(titles[index-1])
    plt.xlabel('k')
    plt.ylim(min(min(meanstats[:,index]), min(medianstats[:,index]))-timedelta, max(max(meanstats[:,index]), max(medianstats[:,index]))+timedelta)
    plt.savefig(outputfilenames[index-1],format='pdf')

if __name__=='__main__':
  main()
