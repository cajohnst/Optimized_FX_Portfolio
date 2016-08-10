import pandas as pd
import numpy as np
import pandas.io.data as web
from pandas.io.data import DataReader
from pandas import Series, DataFrame
import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers, matrix

import pdb

def main():
	currency_list = ['DEXMXUS', 'DEXCAUS', 'DEXUSNZ', 'DEXHKUS', 'DEXJPUS', 'DEXSIUS', 'DEXUSUK', 'DEXSFUS', 'DEXUSAL', 'DEXUSEU']
	num_days = 365
	#Compute exponential weighted moving average returns with shift delay
	shift = 1
	#Compute EWMA returns with shift operator shift
	returns_table = get_currency_data(currency_list, num_days, shift)
	# print returns_table

	#rollover_table = rollover_scraper.generate_rollover(currency_list)
	returns_mean, returns_var, returns_cov = calc_mean_var_cov(returns_table)

	# For simplicity, assume fixed interest rate
	interest_rate = 0.0225/365.
	# Minimum desired return
	rmin = 0.05/365

	# Variable Initialization
	today= datetime.date.today()
	start_date = today - timedelta(num_days)

	sol = MarkowitzOpt(returns_table, returns_mean, returns_var, returns_cov, interest_rate, rmin)

	return sol


	# INDEX = returns_table.index
	# START_INDEX = INDEX.get_loc(start_date)
	# END_DATE = INDEX[-1]
	# END_INDEX = INDEX.get_loc(END_DATE)
	# DATE_INDEX_iter = START_INDEX
	# currency_list.append('RiskFree')
	# DISTRIBUTION = DataFrame(index=currency_list)
	# RETURNS = Series(index=INDEX)
	# # Start Value
	# TOTAL_VALUE = 1.0
	# RETURNS[INDEX[DATE_INDEX_iter]] = TOTAL_VALUE

	# while DATE_INDEX_iter + shift < END_INDEX:
	# 	DATEiter = INDEX[DATE_INDEX_iter]
	# 	# print DATEiter

	# 	xsol = MarkowitzOpt(returns_mean.ix[DATEiter],
	# 						returns_var.ix[DATEiter],
	# 						returns_cov.ix[DATEiter],interest_rate,rmin)

	# 	dist_sum = xsol.sum()
	# 	DISTRIBUTION[DATEiter.strftime('%Y-%m-%d')] = xsol

	# 	DATEiter2 = INDEX[DATE_INDEX_iter+shift]
	# 	temp1 = price.ix[DATEiter2]/price.ix[DATEiter]
	# 	temp1.ix[currency_list[-1]] = interest_rate+1
	# 	temp2 = Series(xsol.ravel(),index=currency_list)
	# 	TOTAL_VALUE = np.sum(TOTAL_VALUE*temp2*temp1)

	# 	# Increase Date
	# 	DATE_INDEX_iter += shift
	# 	#print 'Date:' + str(INDEX[DATE_INDEX_iter])
	# 	RETURNS[INDEX[DATE_INDEX_iter]] = TOTAL_VALUE

	# # Remove dates that there are no trades from returns
	# RETURNS = RETURNS[np.isfinite(RETURNS)]

	# temp3 = DISTRIBUTION.T
	# # To prevent cramped figure, only plotting last 10 periods
	# ax = temp3.ix[-10:].plot(kind='bar',stacked=True)
	# plt.ylim([0,1])
	# plt.xlabel('Date')
	# plt.ylabel('Distribution')
	# plt.title('Distribution vs. Time')
	# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	# plt.figure()
	# RETURNS.plot()
	# plt.xlabel('Date')
	# plt.ylabel('Portolio Returns')
	# plt.title('Portfolio Returns vs. Time')

	# plt.show()

  
def get_currency_data(currency_list, num_days, shift):
	# Calculate dates
	today = datetime.date.today()
	start_date = today - timedelta(num_days)

	# Initialize data table
	data_table = None
	# Run through currencies, first assignment is initialized
	# Anything past first currency is joined into table
	for currency in currency_list:
		current_column = DataReader(currency, 'fred', start_date, today)
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

def calc_mean_var_cov(returns_table):
	# Specify filter "length"
	returns_mean = returns_table.mean()
	returns_var = returns_table.var()
	#Compute covariances
	returns_cov = returns_table.cov()
	return returns_mean, returns_var, returns_cov

def MarkowitzOpt(returns_table, returns_mean,returns_var,returns_cov,irate,rmin):
	# Number of currency pairs
	num_currencies = returns_mean.size
	# Number of positions
	# Additional position for risk free rate
	num_positions = num_currencies + 1
	# mean return vector
	pbar = returns_mean.copy(deep=True)
	pbar['RT'] = irate
	

	# Ensure feasability Code
	if(pbar.max() < rmin):
		rmin_constraint = irate
	else:
		rmin_constraint = rmin;
	
	sigma = returns_table.copy(deep=True)
	sigma['RT'] = irate
	sigma = sigma.cov()
	sigma = matrix(sigma.as_matrix())

	# Generate G matrix and h vector for inequality constraints
	G = matrix(0.0,(num_positions+1,num_positions))
	h = matrix(0.0,(num_positions+1,1))
	h[-1] = -rmin_constraint
	for i in np.arange(num_positions):
		G[i,i] = -1
	G[-1,:] = -pbar
	# Generate p matrix and b vector for equality constraints
	p = matrix(1.0,(1,num_positions))
	b = matrix(1.0)
	q = matrix(0.0,(num_positions,1))

	# Run convex optimization program
	solvers.options['show_progress'] = False
	sol=solvers.qp(sigma,q,G,h,p,b)
	# Solution
	xsol = np.array(sol['x'])
	dist_sum = xsol.sum()

	return xsol

if __name__ == "__main__":
	main()