import pandas as pd
import numpy as np
import pandas.io.data as web
# from pandas.io.data import DataReader
from pandas import Series, DataFrame
import datetime
from datetime import date, timedelta
import quandl as qdl
import cvxopt as opt
from cvxopt import blas, solvers, matrix
import rollover_google_sheet 
# import xlsxwriter

def main():
	currency_list = get_currency_list()
	num_days = 365
	#Compute returns with shift delay
	shift = 1
	#Compute returns
	returns_table_1, returns_table_2 = get_currency_data(currency_list_quandl, currency_list_fred, num_days, shift)

	print returns_table_1
	print returns_table_2
	
	difference_table = returns_table_1 - returns_table_2

	print difference_table

	returns_table_1.to_csv('returns_table_1.txt')
	returns_table_2.to_csv('returns_table_2.txt')
	difference_table.to_csv('difference_table.txt')

	# # For simplicity, assume fixed interest rate
	# interest_rate = 0.005/365.

	# # Minimum desired return
	# rmin = 0.25/365

	# rollover_table = rollover_google_sheet.pull_data(num_days)

	# rollover_table = rollover_table / 10000

	# merge_table = merge_tables(returns_table, rollover_table)
	# merge_table = merge_table.dropna()

	# # Input Leverage
	# leverage = 10/1
	# # Return the returns table with leverage
	# merge_table = merge_table * leverage

	# merge_table['RF'] = interest_rate

	# # Calculate the mean, variance, and covariances of the returns table
	# returns_mean, returns_var, returns_cov = calc_mean_var_cov(merge_table)

	# # writer= pd.ExcelWriter('sample_returns_test.xlsx', engine= 'xlsxwriter')
	# # merge_table.to_excel(writer, 'sheet1')
	# # writer.save()

	# # Compute minimum variance portfolio to match desired expected return, print optimal portfolio weights
	# sol = MarkowitzOpt(merge_table, returns_mean, returns_var, returns_cov, interest_rate, rmin)
	# return sol, merge_table

def get_currency_list():
	currency_list_fred = ['DEXMXUS', 'DEXCAUS', 'DEXUSNZ', 'DEXHKUS', 'DEXJPUS', 'DEXSIUS', 'DEXUSUK', 'DEXSFUS', 'DEXUSAL', 'DEXUSEU']
	currency_list_quandl = ['CURRFX/MXNUSD.1', 'CURRFX/USDCAD.1', 'CURRFX/NZDUSD.1', 'CURRFX/USDHKD.1', 'CURRFX/USDJPY.1', 'CURRFX/USDSGD.1', 'CURRFX/GBPUSD.1', 'CURRFX/USDZAR.1', 'CURRFX/AUDUSD.1', 'CURRFX/EURUSD.1']
	return currency_list_fred, currency_list_quandl

def merge_tables(returns_table, rollover_table):
	merge_table = pd.DataFrame(columns=rollover_table.columns)
	for index, column in enumerate(returns_table):
		merge_table[merge_table.columns[(index * 2)]] = -1 * returns_table.ix[:,index]
		merge_table[merge_table.columns[(index * 2) + 1]] = returns_table.ix[:,index]
	merge_table = merge_table + rollover_table
	return merge_table
  
def get_currency_data(currency_list_quandl, currency_list_fred, num_days, shift):
	# Calculate dates
	end_date = datetime.date.today()
	start_date = end_date - timedelta(num_days)

	# Initialize data table
	data_table = None
	# Run through currencies, first assignment is initialized
	# Anything past first currency is joined into table
	for currency in currency_list_quandl:
		current_column = qdl.get(currency, start_date= start_date, end_date= end_date)
		if data_table is None:
			data_table = current_column
		else:
			data_table = data_table.join(current_column, how= 'left', rsuffix= ' ')

	data_table.columns = ['MXN/USD', 'USD/CAD', 'NZD/USD', 'USD/HKD', 'USD/JPY', 'USD/SGD', 'GBP/USD', 'USD/ZAR', 'AUD/USD', 'EUR/USD']

	# Convert price data to returns and delete NaNs
	returns_table_quandl = data_table.pct_change(periods= shift).dropna()
	# Specify number of days to shift

	returns_table_quandl.drop(returns_table_quandl.index[:1], inplace=True)

		# Initialize data table
	data_table_fred = None
	# Run through currencies, first assignment is initialized
	# Anything past first currency is joined into table
	for currency in currency_list_fred:
		current_column = DataReader(currency, 'fred', start_date, today)
		if data_table_fred is None:
			data_table_fred = current_column
		else:
			data_table_fred = data_table.join(current_column)

	# Convert price data to returns and delete NaNs
	returns_table_fred = data_table_fred.pct_change(periods= shift).dropna()
	# Specify number of days to shift

	returns_table_fred.drop(returns_table_fred.index[:1], inplace=True)
	
	return returns_table_quandl, returns_table_fred

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
	num_positions = num_currencies
	# mean return vector
	pbar = returns_mean.copy(deep=True)

	# Ensure feasability Code
	if(pbar.max() < rmin):
		rmin_constraint = irate
	else:
		rmin_constraint = rmin;
	
	sigma = returns_table.copy(deep=True)
	sigma = sigma.cov()
	sigma = matrix(sigma.as_matrix())

	# Generate G matrix and h vector for inequality constraints
	G = matrix(0.0,(num_positions,num_positions))
	h = matrix(0.0,(num_positions,1))
	h[-1] = -rmin_constraint
	for i in np.arange(num_positions):
		G[i,i] = -1
	G[-1,:] = -pbar

	print h
	print G

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
	print xsol

	return xsol


if __name__ == "__main__":
	main()