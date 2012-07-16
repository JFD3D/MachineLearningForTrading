#Implements KNN Learner
#Author: Arvind Sundararajan
#
import numpy as np

class KNNLearner:
  def __init__(self, k, method = 'mean'):
    self.XData = np.zeros((1,2), dtype = 'float')
    self.YData = np.zeros((1,1), dtype = 'float')
    self.k = k
    self.method = method.lower()

  def addEvidence(self, Xtrain, Ytrain):
    self.XData = Xtrain.copy()
    self.YData = Ytrain.copy()

  def query(self, Xtest):
    Y = np.zeros((Xtest.shape[0], 1), dtype = 'float')
    dist = np.zeros((self.XData.shape[0], 1), dtype = 'float')
    
    for index in range(Xtest.shape[0]):
      dist = (self.XData[:,0] - Xtest[index, 0])**2 + (self.XData[:,1] - Xtest[index, 1])**2
      kneighbours = [self.YData[neighbourindex] for neighbourindex in np.argsort(dist)[:self.k]]
      if self.method == 'median':
        Y[index] = np.median(kneighbours)
      else: #default to 'mean'
        Y[index] = np.mean(kneighbours)
    
    return Y
