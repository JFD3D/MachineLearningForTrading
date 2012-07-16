#Implements Classifier1
#Author: Arvind Sundararajan
#

import sys
import numpy as np
import re as re
import os

def GetWordCountForSingleDocument(filename, MAXV):
  wordcounts = np.zeros(MAXV)
  elementlist = open(filename, 'rU').read()
  words = re.findall(r'[a-z]+', str.lower(elementlist))

  for word in words:
    hashvalue = hash(word) % MAXV
    wordcounts[hashvalue] = wordcounts[hashvalue] + 1

  return wordcounts
    
def GetWordCountsForListOfDocuments(filelistarg, MAXV):
  wordcounts = np.zeros(MAXV)
  #read goodlist.txt
  filelist = open(filelistarg, 'rU').read()
  for filename in filelist.split():
    wordcounts = wordcounts + GetWordCountForSingleDocument(filename, MAXV)
    
  return wordcounts
      
def main(sysargv):
  MAXV = 1000
  goodlist = GetWordCountsForListOfDocuments(sysargv[0], MAXV)
  badlist = GetWordCountsForListOfDocuments(sysargv[1], MAXV)
  
  goodp = goodlist / sum(goodlist)
  badp = badlist / sum(badlist)
  
  weights = goodp  - badp
  weights = weights.astype('float');
  sump = goodp + badp
  sump[sump == 0] = 1
  sump = sump.astype('float');
  weights = np.divide(weights, sump)
        
  totaldocs = 0
  numcorrect = 0
  istrueoutputavailable = 0
  
  filelist = open(sysargv[2], 'rU').read()
  for filename1 in filelist.split():
    if filename1.find('\\') != -1:
      filenamesplit = filename1.split('\\')
    else:
      filenamesplit = filename1.split('/')
    
    filename = filenamesplit[-2] + '/' + filenamesplit[-1]
    print('file: %s'%(filename))
    
    totaldocs = totaldocs + 1
    wordcounts = GetWordCountForSingleDocument(filename, MAXV)
    truelabel = 0
    if str.lower(filenamesplit[-2]).find('good') != -1:
      truelabel = 1
    elif str.lower(filenamesplit[-2]).find('bad') != -1:
      truelabel = -1
      
    if truelabel != 0:
      #print filename
      istrueoutputavailable = 1

    score = sum(np.multiply(weights, wordcounts))
    if score > 0:
      print 'class: good'
      if truelabel == 1:
        numcorrect = numcorrect + 1
    elif score == 0:
      print 'class: neutral'
    else:
      print 'class: bad'
      if truelabel == -1:
        numcorrect = numcorrect + 1


  if istrueoutputavailable == 1:
    print('NumCorrect:%d TotalDocs:%d'%(numcorrect, totaldocs))

if __name__=='__main__':
  if len(sys.argv) < 4:
    sys.exit('Usage: python %s goodlist.txt badlist.txt testlist.txt'% (sys.argv[0]))

  main(sys.argv[1:])
