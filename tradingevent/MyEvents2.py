#
# Find events
# by Arvind Sundararajan
# October 2011
#
#ARVIND: changed the import of pandas to pandas.stats.moments
import pandas.stats.moments as pa
from qstkutil import DataAccess as da
import numpy as np
import math
import qstkutil.dateutil as du
import datetime as dt
import qstkutil.DataAccess as da

"""
Accepts a list of symbols along with start and end date
Returns the Event Matrix which is a pandas Datamatrix
Event matrix has the following structure :
    |IBM |GOOG|XOM |MSFT| GS | JP |
(d1)|nan |nan | 1  |nan |nan | 1  |
(d2)|nan | 1  |nan |nan |nan |nan |
(d3)| 1  |nan | 1  |nan | 1  |nan |
(d4)|nan |  1 |nan | 1  |nan |nan |
...................................
...................................
Also, d1 = start date
nan = no information about any event.
1 = status bit(positively confirms the event occurence)
"""
# Get the data from the data store
storename = "Norgate" # get data from our daily prices source
# Available field names: open, close, high, low, close, actual_close, volume
closefield = "close"
volumefield = "volume"
window = 10

def findEvents(symbols, startday,endday,verbose=False):
  timeofday=dt.timedelta(hours=16)
  timestamps = du.getNYSEdays(startday,endday,timeofday)
  dataobj = da.DataAccess('Norgate')
  if verbose:
    print __name__ + " reading data"

  adjclose = dataobj.get_data(timestamps, symbols, closefield)
  adjclose = (adjclose.fillna()).fillna(method='backfill')

  adjcloseSPY = dataobj.get_data(timestamps, ['SPY'], closefield)
  adjcloseSPY = (adjcloseSPY.fillna()).fillna(method='backfill')
  
  if verbose:
    print __name__ + " finding events"

  # for symbol in symbols:
  # close[symbol][close[symbol]>= 1.0] = np.NAN
  # for i in range(1,len(close[symbol])):
  # if np.isnan(close[symbol][i-1]) and close[symbol][i] < 1.0 :#(i-1)th was > $1, and (i)th is <$1
  # close[symbol][i] = 1.0 #overwriting the price by the bit
  # close[symbol][close[symbol]< 1.0] = np.NAN

  #print adjclose      
  # Get the 20 day moving avg and moving stddev
  movavg = pa.rolling_mean(adjclose,20,min_periods=20)
  movavgSPY = pa.rolling_mean(adjcloseSPY,20,min_periods=20)
  movstddev = pa.rolling_std(adjclose, 20, min_periods=20)
  movstddevSPY = pa.rolling_std(adjcloseSPY, 20, min_periods=20)
  
  upperband = movavg + 2*movstddev
  upperbandSPY = movavgSPY + 2*movstddevSPY
  lowerband = movavg - 2*movstddev
  lowerbandSPY = movavgSPY - 2*movstddevSPY

  # Compute the bollinger %b indicator for all stocks
  normalizedindicator = 2*(adjclose - movavg)/(upperband - lowerband)
  #print normalizedindicator
  normalizedindicatorSPY = 2*(adjcloseSPY - movavgSPY)/(upperbandSPY - lowerbandSPY)
  #print normalizedindicatorSPY
  #bandwidth = (upperband - lowerband)/movavg
  #print bandwidth
  #print upperband
  # Compute the event matrix as follows:
  # Set periods of low volatility to 1
  # In from the period of low volatility to the period of say, 15 days, following low volatility 
  # if the stock price breaks above the upper band, there is a surge. this is a positive event. Set this event to 2
  # Finally, set all events other than 2 to NaN. Then, set all 2's to 1
  eventMatrix = adjclose.copy()
  
  for symbol in symbols:
    for row in range(len(adjclose[:][symbol])):
      eventMatrix[symbol][row] = np.NAN
      if normalizedindicator[symbol][row] - normalizedindicatorSPY['SPY'][row] >= 0.5:
        eventMatrix[symbol][row] = 1

  return eventMatrix
