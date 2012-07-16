#Code for Feature Selection
#Author: Arvind Sundararajan
#
import qstkutil.dateutil as du
import qstkutil.DataAccess as da
import qstklearn.kdtknn as knn
import qstkfeat.featutil as ftu
from qstkfeat.features import *
from qstkfeat.featutil import *

import numpy as np
import datetime as dt
import pandas as pand
  
def main():
  #symbols = np.loadtxt('./Examples/Features/symbols.txt',dtype='S10',comments='#')
  symbols = ['AA', 'AXP', 'BA', 'BAC', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE', 'HD', 'HPQ', 'IBM', 'INTC', 'JNJ', \
  'JPM', 'KFT', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'PFE', 'PG', 'T', 'TRV', 'UTX', 'VZ', 'WMT', 'XOM'  ]
  #symbols = ['XOM']
  #This is the start and end dates for the entire train and test data combined
  alldatastartday = dt.datetime(2007,1,1)
  alldataendday = dt.datetime(2010,6,30)
  timeofday=dt.timedelta(hours=16)
  timestamps = du.getNYSEdays(alldatastartday,alldataendday,timeofday)
  dataobj = da.DataAccess('Norgate')
  voldata = dataobj.get_data(timestamps, symbols, "volume",verbose=True)
  voldata = (voldata.fillna()).fillna(method='backfill')
  close = dataobj.get_data(timestamps, symbols, "close",verbose=True)
  close = (close.fillna()).fillna(method='backfill')
  
  featureList = [featMA, featMA, featRSI, featRSI, featDrawDown, featRunUp, featVolumeDelta, featVolumeDelta, featAroon, classFutRet]
  featureListArgs = [{'lLookback':10, 'bRel':True},\
                      {'lLookback':20},\
                      {'lLookback':10},\
                      {'lLookback':20},\
                      {},\
                      {},\
                      {'lLookback':10},\
                      {'lLookback':20},\
                      {'bDown':False},\
                      {'lLookforward':5}]

  #print 'Applying Features'
  #
  # John Cornwell's featuretest.py was consulted for figuring out the syntax of ftu.applyFeatures() methods and ftu.stackSyms() methods
  #
  allfeatureValues = ftu.applyFeatures(close, voldata, featureList, featureListArgs)
  
  trainstartday = dt.datetime(2007,1,1)
  trainendday = dt.datetime(2009,12,31)
  traintimestamps = du.getNYSEdays(trainstartday,trainendday,timeofday)
  #print 'Stack Syms for Training'
  trainingData = ftu.stackSyms(allfeatureValues, traintimestamps[0], traintimestamps[-1])
  #print 'Norm Features for Training'
  scaleshiftvalues = ftu.normFeatures( trainingData, -1.0, 1.0, False )
  
  teststartday = dt.datetime(2010,1,1)
  testendday = dt.datetime(2010,6,30)
  testtimestamps = du.getNYSEdays(teststartday,testendday,timeofday)
  #print 'Stack Syms for Test'
  testData = ftu.stackSyms(allfeatureValues, testtimestamps[0], testtimestamps[-1])
  #print 'Norm Features for Test'
  ftu.normQuery(testData[:,:-1], scaleshiftvalues)


  NUMFEATURES = 9
  bestFeatureIndices = []
  bestCorrelation = 0.0
  
  fid = open("output.txt", 'w')
  
  for iteration in range(NUMFEATURES):
    nextFeatureIndexToAdd = -1

    for featureIndex in range(NUMFEATURES):
      
      if featureIndex not in bestFeatureIndices:
      
        bestFeatureIndices.append(featureIndex) 
      
        fid.write('testing feature set '+str(bestFeatureIndices)+'\n')
        print('testing feature set '+str(bestFeatureIndices))
        
        bestFeatureIndices.append(9)
        curTrainingData = trainingData[:,bestFeatureIndices]
        curTestData = testData[:,bestFeatureIndices]
        bestFeatureIndices.remove(9)

        kdtlearner = knn.kdtknn(5, 'mean', leafsize=100)
        kdtlearner.addEvidence(curTrainingData[:,:-1], curTrainingData[:,-1])
        testEstimatedValues = kdtlearner.query(curTestData[:,:-1])
        testcorrelation = np.corrcoef(testEstimatedValues.T, curTestData[:,-1].T)  
        curCorrelation = testcorrelation[0,1]
        
        fid.write('corr coef = %.4f\n'%(curCorrelation))
        print('corr coef = %.4f'%(curCorrelation))
        
        if curCorrelation > bestCorrelation:
          nextFeatureIndexToAdd = featureIndex
          bestCorrelation = curCorrelation
          
        bestFeatureIndices.remove(featureIndex)
        
    if nextFeatureIndexToAdd >= 0:
      bestFeatureIndices.append(nextFeatureIndexToAdd)
    else:
      break
  
  fid.write('best feature set is '+str(bestFeatureIndices)+'\n')
  print('best feature set is '+str(bestFeatureIndices))
  fid.write('corr coef = %.4f'%(bestCorrelation)+'\n')
  print('corr coef = %.4f'%(bestCorrelation))
  fid.close()

if __name__=='__main__':
  main()
