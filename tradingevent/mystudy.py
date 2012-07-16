#
# Example use of the event profiler
#
#ARVIND: changed this to MyEvents
import MyEvents as ev
import datetime as dt
import EventProfiler as ep
import numpy as np
#symbols = ['BFRE','ATCS','RSERF','GDNEF','LAST','ATTUF','JBFCF','CYVA','SPF','XPO','EHECF','TEMO','AOLS','CSNT','REMI','GLRP','AIFLY','BEE','DJRT','CHSTF','AICAF']
symbols = np.loadtxt('symbol-set2-wohyphen.txt',dtype='S10',comments='#')
#ARVIND: use ABX for testing
#symbols = ['ABX']
#symbols = symbols[0:20]
#ARVIND: changed the start and end day to include data during the non-recession period
startday = dt.datetime(2007,1,1)
#startday = dt.datetime(2004,1,1)
endday = dt.datetime(2007,12,31)
eventMatrix = ev.findEvents(symbols,startday,endday,verbose=True)
#print eventMatrix
#print 'eventMatrix call ended'
#print len(eventMatrix)
eventProfiler = ep.EventProfiler(eventMatrix,startday,endday,
        lookback_days=20,lookforward_days=20,verbose=True)

#ARVIND: i need plotMarketNeutral to be False
eventProfiler.study(filename="MyEventStudy.pdf",\
	plotErrorBars=True,\
	plotMarketNeutral=False,\
        plotEvents=False,\
	marketSymbol='SPY')
