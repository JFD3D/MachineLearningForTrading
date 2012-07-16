import qstkutil.dateutil as du
import qstkutil.tsutil as tsu
import qstkutil.DataAccess as da
import datetime as dt
import matplotlib.pyplot as plt
from pylab import *

#
# Prepare to read the data
#
symbols = ["AAPL","GLD","GOOG","SPY","XOM"]
#
#for testing wth the graphs on wiki, uncomment the below two datetimes
#and comment the two lines following these two
#for actual submission, do the reverse
#
#startday = dt.datetime(2008,1,1)
#endday = dt.datetime(2009,12,31)
startday = dt.datetime(2007,1,1)
endday = dt.datetime(2010,12,31)
timeofday=dt.timedelta(hours=16)
timestamps = du.getNYSEdays(startday,endday,timeofday)

dataobj = da.DataAccess('Norgate')
voldata = dataobj.get_data(timestamps, symbols, "volume",verbose=True)
close = dataobj.get_data(timestamps, symbols, "close",verbose=True)
actualclose = dataobj.get_data(timestamps, symbols, "actual_close",verbose=True)

#
# Plot the adjusted close data
#
plt.clf()
newtimestamps = close.index
pricedat = close.values # pull the 2D ndarray out of the pandas object
plt.plot(newtimestamps,pricedat)
plt.legend(symbols)
plt.ylabel('Adjusted Close')
plt.xlabel('Date')
savefig('adjustedclose.pdf',format='pdf')

#
# Plot the normalized closing data
#
plt.clf()
normdat = pricedat/pricedat[0,:]
plt.plot(newtimestamps,normdat)
plt.legend(symbols)
plt.ylabel('Normalized Close')
plt.xlabel('Date')
savefig('normalized.pdf',format='pdf')

#
# Plot daily returns
#
plt.clf()
plt.cla()
tsu.returnize0(normdat)
plt.plot(newtimestamps[0:50],normdat[0:50,3]) # SPY 50 days
plt.plot(newtimestamps[0:50],normdat[0:50,4]) # XOM 50 days
plt.axhline(y=0,color='r')
plt.legend(['SPY','XOM'])
plt.ylabel('Daily Returns')
plt.xlabel('Date')
savefig('rets.pdf',format='pdf')

#
# Scatter plat
#
plt.clf()
plt.cla()
plt.scatter(normdat[:,3],normdat[:,4],c='blue') # SPY v XOM
plt.ylabel('XOM')
plt.xlabel('SPY')
savefig('scatter.pdf',format='pdf')

#
#Cumulative Daily Returns
#
plt.clf()
plt.cla()
daily_cum_ret=1+normdat
daily_cum_ret[1:,]*= daily_cum_ret[0:-1,]
plt.plot(newtimestamps,daily_cum_ret)
plt.legend(symbols)
plt.ylabel('Daily Cumulative Returns')
plt.xlabel('Date')
savefig('cumulative.pdf',format='pdf')


#
#Combining Daily Returns to Estimate Portfolio Returns
#
plt.clf()
plt.cla()
portfolio_daily_ret=daily_cum_ret
newportfolio=zeros(normdat.shape[0], dtype=normdat.dtype)
newportfolio=expand_dims(newportfolio, axis=1)
portfolio_daily_ret=concatenate((portfolio_daily_ret, newportfolio), axis=1)
portfolio_daily_ret[:,5]= 0.75*normdat[:,1] + 0.25*normdat[:,3]
portfolio_daily_ret[:,5]+=1
portfolio_daily_ret[1:,5]*=portfolio_daily_ret[0:-1,5]
#print portfolio_daily_ret[1:10,]
plt.plot(newtimestamps,portfolio_daily_ret[:,3]) # SPY
plt.plot(newtimestamps,portfolio_daily_ret[:,1]) # GLD
plt.plot(newtimestamps,portfolio_daily_ret[:,5]) # New Portfolio
plt.ylim(0.4, 2.4)
plt.axhline(y=0,color='r')
plt.legend(['SPY','GLD', 'New Portfolio'])
plt.ylabel('Adjusted Close')
plt.xlabel('Date')
savefig('combined.pdf',format='pdf')

#
# Line fit to Daily Returns
#
plt.clf()
plt.cla()
plt.scatter(normdat[:,3],normdat[:,4],c='blue') # SPY v XOM
corr=corrcoef([normdat[:,3].T, normdat[:,4].T])
#print corr

#
#Uncomment the following to obtain the line fit using polyfit, polyval
#
polydim = 1
polycoeff=polyfit(normdat[:,3], normdat[:,4], polydim)
spy_data=normdat[:,3]
spy_data.sort()
polyfit_xom_data=polyval(polycoeff, spy_data)
slope=polycoeff[0]
plt.plot(spy_data, polyfit_xom_data, c='red')

#
#Uncomment the following to obtain the line fit using the inbuilt least squares regression
#
#spy_data=normdat[:,3]
#xom_data=normdat[:,4]
#spy_data_linreg=vstack([spy_data, ones(len(spy_data))]).T
#slope, intercept = linalg.lstsq(spy_data_linreg, xom_data)[0]
#spy_data.sort()
#plt.plot(spy_data, slope*spy_data+intercept, c='red')

legend_for_linefit = 'corr = '+str(corr[0,1])[:5] + ', slope = ' + str(slope)[:5]
legend_for_scatter='SPYvXOM'
plt.legend([legend_for_linefit, legend_for_scatter])
plt.ylabel('XOM')
plt.xlabel('SPY')
savefig('line.pdf',format='pdf')
