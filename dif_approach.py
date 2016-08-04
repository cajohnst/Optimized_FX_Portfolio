import pandas as pd
import numpy as np
import pandas.io.data as web
from pandas.io.data import DataReader
from pandas import Series, DataFrame
import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers

def main():
	currency_list = ['DEXUSEU', 'DEXUSUK', 'DEXUSAL', 'DEXSFUS', 'DEXUSNZ', 'DEXSIUS', 'DEXMXUS', 'DEXJPUS', 'DEXCAUS', 'DEXHKUS']
	num_days = 365
    #Compute exponential weighted moving average returns with shift delay
	shift = 5
    #Compute EWMA returns with shift operator shift
	returns_table = get_currency_data(currency_list, num_days, shift)

	#rollover_table = rollover_scraper.generate_rollover(currency_list)

	mean_var_cov = calc_mean_var_cov(returns_table, shift, currency_list)
	print returns_table
	# print shift_returns_mean
	# print shift_returns_var
	# print CovSeq

    # For simplicity, assume fixed interest rate
	interest_rate = 0.0225/365.
	# Minimum desired return
	rmin = 0.05/365

    # Variable Initialization
	today= datetime.date.today()
	duration = today - timedelta(num_days + 10)
	INDEX = returns_table.index
	START_INDEX = INDEX.get_loc(duration)
	END_DATE = INDEX[-1]
	END_INDEX = INDEX.get_loc(END_DATE)
	DATE_INDEX_iter = START_INDEX
	currency_list.append('RiskFree')
	DISTRIBUTION = DataFrame(index=currency_list)
	RETURNS = Series(index=INDEX)
	# Start Value
	TOTAL_VALUE = 1.0
	RETURNS[INDEX[DATE_INDEX_iter]] = TOTAL_VALUE

	while DATE_INDEX_iter + shift < END_INDEX:
		DATEiter = INDEX[DATE_INDEX_iter]
		# print DATEiter

		xsol = MarkowitzOpt(shift_returns_mean.ix[DATEiter],
	                        shift_returns_var.ix[DATEiter],
	                        CovSeq.ix[DATEiter],interest_rate,rmin)

		dist_sum = xsol.sum()
		DISTRIBUTION[DATEiter.strftime('%Y-%m-%d')] = xsol

		DATEiter2 = INDEX[DATE_INDEX_iter+shift]
		temp1 = price.ix[DATEiter2]/price.ix[DATEiter]
		temp1.ix[currency_list[-1]] = interest_rate+1
		temp2 = Series(xsol.ravel(),index=currency_list)
		TOTAL_VALUE = np.sum(TOTAL_VALUE*temp2*temp1)

		# Increase Date
		DATE_INDEX_iter += shift
		#print 'Date:' + str(INDEX[DATE_INDEX_iter])
		RETURNS[INDEX[DATE_INDEX_iter]] = TOTAL_VALUE

	# Remove dates that there are no trades from returns
	RETURNS = RETURNS[np.isfinite(RETURNS)]

	temp3 = DISTRIBUTION.T
	# To prevent cramped figure, only plotting last 10 periods
	ax = temp3.ix[-10:].plot(kind='bar',stacked=True)
	plt.ylim([0,1])
	plt.xlabel('Date')
	plt.ylabel('Distribution')
	plt.title('Distribution vs. Time')
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	plt.figure()
	RETURNS.plot()
	plt.xlabel('Date')
	plt.ylabel('Portolio Returns')
	plt.title('Portfolio Returns vs. Time')

	plt.show()
  
def get_currency_data(currency_list, num_days, shift):
    # Calculate dates
    today = datetime.date.today()
    duration = today - timedelta(num_days)

    # Initialize data table
    data_table = None
    # Run through currencies, first assignment is initialized
    # Anything past first currency is joined into table
    for currency in currency_list:
        current_column = DataReader(currency, 'fred', duration, today)
        if data_table is None:
            data_table = current_column
        else:
            data_table = data_table.join(current_column)

    # Convert price data to returns and delete NaNs
    returns_table = data_table.pct_change(periods= shift).dropna()
    # Specify number of days to shift

    returns_table.drop(returns_table.index[:1], inplace=True)
    return returns_table
    
""" Optimization Step
Maximize interest rate return while minimizing variance in day to day moves.  This does not take moves in exchange rate into account currently.
Rollover table: Columns: Pairs, Rows: Short/Long
Eventually, will take average of 60-100 days of rollovers
    
#The following is borrowed from this link: http://blog.quantopian.com/markowitz-portfolio-optimization-2/
"""
def calc_mean_var_cov(returns_table, shift, currency_list):
	# Specify filter "length"
	filter_len = shift
	shift_returns_mean = DataFrame.ewm(returns_table, ignore_na=False, span= filter_len, min_periods=0, adjust=True).mean(other= None)
	shift_returns_var = DataFrame.ewm(returns_table, ignore_na=False, span= filter_len, min_periods=0, adjust=True).var(bias=False)

	#Compute covariances
	NumCurrencies = len(currency_list)
	CovSeq = pd.DataFrame()
	for FirstPair in np.arange(NumCurrencies-1):
		for SecondPair in np.arange(FirstPair+1,NumCurrencies):
			ColumnTitle = currency_list[FirstPair] + '-' + currency_list[SecondPair]
			#convert returns table to series
			CovSeq[ColumnTitle] = Series.ewm(returns_table, ignore_na=False, span= filter_len, min_periods=0, adjust=True).cov(other=None ,bias=False)
	return shift_returns_mean 
	return shift_returns_var
	return CovSeq

def MarkowitzOpt(meanvec,varvec,covvec,irate,rmin):

	# Number of positions
	# Additional position for risk free rate
	numPOS = meanvec.size+1
	# Number of currency pairs
	Num_currencies = meanvec.size
	# mean return vector
	pbar = matrix(irate,(1,numPOS))
	pbar[:numPOS-1]=matrix(meanvec)

	# Ensure feasability Code
	pbar2 = np.array(pbar)
	if(pbar2.max() < rmin):
		rmin_constraint = irate
	else:
		rmin_constraint = rmin;

	counter = 0
	SIGMA = matrix(0.0,(numPOS,numPOS))
	for i in np.arange(NumStocks):
		for j in np.arange(i,NumStocks):
			if i == j:
				SIGMA[i,j] = varvec[i]
			else:
				SIGMA[i,j] = covvec[counter]
				SIGMA[j,i] = SIGMA[i,j]
				counter+=1

	# Generate G matrix and h vector for inequality constraints
	G = matrix(0.0,(numPOS+1,numPOS))
	h = matrix(0.0,(numPOS+1,1))
	h[-1] = -rmin_constraint
	for i in np.arange(numPOS):
		G[i,i] = -1
	G[-1,:] = -pbar
	# Generate p matrix and b vector for equality constraints
	p = matrix(1.0,(1,numPOS))
	b = matrix(1.0)
	q = matrix(0.0,(numPOS,1))

	# Run convex optimization program
	solvers.options['show_progress'] = False
	sol=solvers.qp(SIGMA,q,G,h,p,b)
	# Solution
	xsol = np.array(sol['x'])
	dist_sum = xsol.sum()

	return xsol

if __name__ == "__main__":
	main()