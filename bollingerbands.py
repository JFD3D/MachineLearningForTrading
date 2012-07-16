import qstkutil.dateutil as du
import qstkutil.tsutil as tsu
import qstkutil.DataAccess as da
import datetime as dt
import matplotlib.pyplot as plt
import pandas.stats.moments as pa
from pylab import *

#
# Prepare to read the data
#

#
#Uncomment this to test with the picture on the wiki
#
#startday = dt.datetime(2010,1,1)
#endday = dt.datetime(2010,10,1)
#stock='VZ'

startday = dt.datetime(2009,1,1)
endday = dt.datetime(2010,1,1)
stock='IBM'
symbols = [stock]
timeofday=dt.timedelta(hours=16)
timestamps = du.getNYSEdays(startday,endday,timeofday)

dataobj = da.DataAccess('Norgate')
adjclose = dataobj.get_data(timestamps, symbols, "close")

adjclose = adjclose.fillna()
adjclose = adjclose.fillna(method='backfill')

# Get the 20 day moving avg and moving stddev
movavg = pa.rolling_mean(adjclose,20,min_periods=20)
movstddev = pa.rolling_std(adjclose, 20, min_periods=20)

# Compute the upper and lower bollinger bands
upperband = movavg + 2*movstddev
lowerband = movavg - 2*movstddev

# Plot the adjclose, movingavg, upper and lower bollinger bands
plt.clf()

plt.plot(adjclose.index,adjclose[stock].values)
plt.plot(adjclose.index,movavg[stock].values)
plt.plot(adjclose.index,upperband[stock].values)
plt.plot(adjclose.index,lowerband[stock].values)
plt.xlim(adjclose.index[0], adjclose.index[len(adjclose.index)-1])

plt.legend(['IBM','Moving Avg.','Upper Bollinger Band','Lower Bollinger Band'], loc='upper left')
plt.ylabel('Adjusted Close')
plt.xlabel('Date')
savefig("bollingerplot1.pdf", format='pdf')

# Normalize the upper and lower bollinger bands in range [-1,+1]
normalizedupperband = upperband - movavg - 2*movstddev + 1
normalizedlowerband = lowerband - movavg + 2*movstddev - 1

# Normalize the Bollinger %b indicator in the range [-1,+1]
normalizedindicator = 2*(adjclose - movavg)/(upperband - lowerband)

# plot the normalized bollinger upper band, lower band and indicator
plt.clf()
plt.plot(adjclose.index,normalizedindicator[stock].values)
plt.plot(adjclose.index,normalizedupperband[stock].values)
plt.plot(adjclose.index,normalizedlowerband[stock].values)
plt.axhline(y=1, color='gray')
plt.axhline(y=-1, color='gray')
plt.xlim(adjclose.index[0], adjclose.index[len(adjclose.index)-1])

#plt.legend(['Normalized Indicator','Normalized Upper Bollinger Band','Normalized Lower Bollinger Band'], loc='upper left')
plt.ylabel('Bollinger Feature')
plt.xlabel('Date')
savefig("bollingerplot2.pdf", format='pdf')


