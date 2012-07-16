#Implements Classifier1
#Author: Arvind Sundararajan
#

import sys
import numpy as np
import re as re
import os
import math

MAXWORDSINVOCABULARY = 100000
idf = np.zeros(MAXWORDSINVOCABULARY, dtype=float)
pwgivengood = np.zeros(MAXWORDSINVOCABULARY, dtype = float)
pwgivenbad = np.zeros(MAXWORDSINVOCABULARY, dtype = float)
word2idmap = {}
id2wordmap = {}

def GetHash(word):
  if word not in word2idmap:
    curvocabsize = len(word2idmap)
    word2idmap[word] = curvocabsize + 1
    id2wordmap[curvocabsize] = word
    
  return word2idmap[word]

def GetWordCountForSingleDocument(filename, istrainingphase, includeunigrams, includebigrams, skipwordlen):
  wordcounts = np.zeros(MAXWORDSINVOCABULARY)
  elementlist = open(filename, 'rU').read()
  words = re.findall(r'[a-z]+', str.lower(elementlist))
  vec = []
  for i in range(len(words)):
    if includeunigrams == 1 and len(words[i]) > skipwordlen:
      hashvalue = GetHash(words[i])
      wordcounts[hashvalue] = wordcounts[hashvalue] + 1
      if istrainingphase == 1 and hashvalue not in vec:
        vec.append(hashvalue)

    if includebigrams == 1 and i > 0 and len(words[i]) > skipwordlen and len(words[i-1]) > skipwordlen:
      hashvalue = GetHash(words[i-1]+' '+words[i])
      wordcounts[hashvalue] = wordcounts[hashvalue] + 1
      if istrainingphase == 1 and hashvalue not in vec:
        vec.append(hashvalue)
        
  if istrainingphase == 1:
    for hashvalue in vec:
        idf[hashvalue] = idf[hashvalue] + 1
        
  return wordcounts
    
def GetWordCountsForListOfDocuments(filelistarg, istrainingphase, includeunigrams, includebigrams, skipwordlen):
  wordcounts = np.zeros(MAXWORDSINVOCABULARY)
  filelist = open(filelistarg, 'rU').read()
  for filename in filelist.split():
    wordcounts = wordcounts + GetWordCountForSingleDocument(filename, istrainingphase, includeunigrams, includebigrams, skipwordlen)
    
  return wordcounts

def BuildNaivesBayesMultinomialModel(filelistarg, isgoodlist, includeunigrams, includebigrams, skipwordlen):
  filelist = open(filelistarg, 'rU').read()
  wordcounts = np.ones(MAXWORDSINVOCABULARY) #Laplace Smoothing for numerator
  denominator = 2 #Laplace smoothing for denominator
  
  for filename in filelist.split():
    curdocwordcounts = GetWordCountForSingleDocument(filename, 1, includeunigrams, includebigrams, skipwordlen)
    denominator = denominator + sum(curdocwordcounts)
    wordcounts = wordcounts + curdocwordcounts
    
  wordcounts = wordcounts.astype('float')
  denominator = float(denominator)
  
  global pwgivengood
  global pwgivenbad
  
  if isgoodlist:
    pwgivengood = wordcounts / denominator
  else:
    pwgivenbad = wordcounts / denominator
  
  return len(filelist.split())
  
def Printpwgivenlabel(isgoodlabel):
  if isgoodlabel:
    print 'pwgivengood'
  else:
    print 'pwgivenbad'
    
  vec = []
  for i in range(MAXWORDSINVOCABULARY):
    s = ''
    if i in id2wordmap:
      s = id2wordmap[i]
      
    if isgoodlabel:
      vec.append((pwgivengood[i], s, i))
    else:
      vec.append((pwgivenbad[i], s, i))
  
  vec.sort()
  vec.reverse()
  for tuple in vec:
    print('Weight:%f Word:%s Id:%d'%(tuple[0], tuple[1], tuple[2]))
    
def main(sysargv, includeunigrams, includebigrams, skipwordlen, includeidfweighting, printoutput):
  
  numgooddocs = BuildNaivesBayesMultinomialModel(sysargv[0], 1, includeunigrams, includebigrams, skipwordlen)
  numbaddocs = BuildNaivesBayesMultinomialModel(sysargv[1], 0, includeunigrams, includebigrams, skipwordlen)
  
  global pwgivengood
  global pwgivenbad

  #Weigh both good and bad by the idfs and renormalize so as to make it a probability
  if includeidfweighting == 1:
    tmpidf = idf.copy()
    tmpidf[tmpidf < 1] = numgooddocs + numbaddocs
    pwgivengood = np.divide(pwgivengood, tmpidf)
    pwgivengood = pwgivengood / sum(pwgivengood)
    pwgivenbad = np.divide(pwgivenbad, tmpidf)
    pwgivenbad = pwgivenbad / sum(pwgivenbad)

  #Printpwgivenlabel(1);
  #Printpwgivenlabel(0);

  pwgivengood = np.log(pwgivengood)
  pwgivenbad = np.log(pwgivenbad)
  
  numgooddocs = float(numgooddocs)
  numbaddocs = float(numbaddocs)
  pgood = numgooddocs / (numgooddocs + numbaddocs)
  pbad = 1.0 - pgood
  if numgooddocs == 0:
    pgood = 1e-10
  if numbaddocs == 0:
    pbad = 1e-10
  
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
    if printoutput == 1: print('file: %s'%(filename))
    
    totaldocs = totaldocs + 1
    wordcounts = GetWordCountForSingleDocument(filename, 0, includeunigrams, includebigrams, skipwordlen)
    truelabel = 0
    if str.lower(filenamesplit[-2]).find('good') != -1:
      truelabel = 1
    elif str.lower(filenamesplit[-2]).find('bad') != -1:
      truelabel = -1
      
    if truelabel != 0:
      istrueoutputavailable = 1

    goodscore = math.log(pgood) + sum(np.multiply(pwgivengood, wordcounts))
    badscore = math.log(pbad) + sum(np.multiply(pwgivenbad, wordcounts))
    
    #print('goodscore:%f badscore:%f'%(goodscore, badscore))
    
    if goodscore > badscore:
      if printoutput == 1: print 'class: good'
      if truelabel == 1: numcorrect = numcorrect + 1
    elif goodscore == badscore:
      if printoutput == 1: print 'class: neutral'
    else:
      if printoutput == 1: print 'class: bad'
      if truelabel == -1: numcorrect = numcorrect + 1


  if istrueoutputavailable == 1:
    print('NumCorrect:%d TotalDocs:%d'%(numcorrect, totaldocs))

  
if __name__=='__main__':
  if len(sys.argv) < 4:
    sys.exit('Usage: python %s goodlist.txt badlist.txt testlist.txt'% (sys.argv[0]))

  includeunigrams = 1
  includebigrams = 1
  skipwordlen = 2
  includeidfweighting = 0
  printoutput = 1

  main(sys.argv[1:], includeunigrams, includebigrams, skipwordlen, includeidfweighting, printoutput)
