import pandas as pd
import numpy as np
import pandas.io.data as web
from pandas.io.data import DataReader
from pandas import Series, DataFrame
import datetime
from datetime import date, timedelta
import cvxopt as opt
from cvxopt import blas, solvers, matrix
import rollover_google_sheet 

def main():
	currency_list = get_currency_list()
	num_days = 365
	#Compute returns with shift delay
	shift = 1
	#Compute returns
	returns_table = get_currency_data(currency_list, num_days, shift)

	''' ***Import and merge rollover table with returns_table here*** '''

	# For simplicity, assume fixed interest rate
	interest_rate = 0.0225/365.

	# Minimum desired return
	rmin = 0.05/365

	# Input Leverage
	leverage = 10/3
	# Return the returns table with leverage
	returns_table = returns_table * leverage

	rollover_table = rollover_google_sheet.pull_data(num_days)

	rollover_table = rollover_table / 10000

	merge_table = merge_tables(returns_table, rollover_table)
	merge_table = merge_table.dropna()
	merge_table['RF'] = interest_rate

	# Calculate the mean, variance, and covariances of the returns table
	returns_mean, returns_var, returns_cov = calc_mean_var_cov(merge_table)

	# Compute minimum variance portfolio to match desired expected return, print optimal portfolio weights
	sol = MarkowitzOpt(merge_table, returns_mean, returns_var, returns_cov, interest_rate, rmin)
	return sol, merge_table

	''' ***Export weights to google spreadsheet to concatenate*** '''

def get_currency_list():
	currency_list = ['DEXMXUS', 'DEXCAUS', 'DEXUSNZ', 'DEXHKUS', 'DEXJPUS', 'DEXSIUS', 'DEXUSUK', 'DEXSFUS', 'DEXUSAL', 'DEXUSEU']
	return currency_list

def merge_tables(returns_table, rollover_table):
	merge_table = pd.DataFrame(columns=rollover_table.columns)
	for index, column in enumerate(returns_table):
		merge_table[merge_table.columns[(index * 2)]] = returns_table.ix[:,index]
		merge_table[merge_table.columns[(index * 2) + 1]] = returns_table.ix[:,index]
	merge_table = merge_table + rollover_table
	return merge_table
  
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

	# Ensure feasability Code
	if(pbar.max() < rmin):
		rmin_constraint = irate
	else:
		rmin_constraint = rmin;
	
	sigma = returns_table.copy(deep=True)
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