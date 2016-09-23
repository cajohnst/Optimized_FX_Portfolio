''' To do: robust covariance matrix, normalize data for cvxopt '''
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import datetime
from datetime import date, timedelta
import quandl as qdl
import cvxopt as opt
from cvxopt import blas, solvers, matrix
import rollover_google_sheet 
import matplotlib.pyplot as plt
import Pull_Data
# from sklearn.covariance import OAS

def main():
	auth_tok = "kz_8e2T7QchJBQ8z_VSi"
	end_date = datetime.date.today()
	np.random.seed(919)
	currency_list = Pull_Data.get_currency_list()
	currency_quandl_list = Pull_Data.get_currency_quandl_list()
	num_days = 500
	# rollover_days = 50
	#Compute returns with shift percentage change delay (daily = 1)
	shift = 1
	#Compute returns
	# currency_table = Pull_Data.get_currency_data(currency_list, currency_quandl_list, num_days, end_date, auth_tok)
	# returns_table = currency_table.pct_change(periods= shift).dropna()
	# returns_table.drop(returns_table.index[:1], inplace=True)

	n_assets = 10

	## NUMBER OF OBSERVATIONS
	n_obs = 500

	return_vec = np.random.randn(n_assets, n_obs)
	pbar = opt.matrix(np.mean(return_vec, axis=1))

	# #For simplicity, assume fixed interest rate
	interest_rate = .02/365
	print interest_rate 

	# # Minimum desired return

	rminimum = 2/365
	print 'R min before function'
	print rminimum

	r_minimum = 2/365
	print 'R min with underscore?'
	print r_minimum

	# rollover_table = rollover_google_sheet.pull_data(num_days)

	# rollover_table = rollover_table / 10000
	# # rollover_table['RF'] = 0

	# # # Input Leverage
	# leverage = 10

	# mean_rollover = np.mean(rollover_table, axis=0)
	# mean_rollover = leverage * opt.matrix(np.append(mean_rollover, np.array(0)))

	# merge_table = merge_tables(returns_table, rollover_table)
	# merge_table = leverage * merge_table.dropna()
	# merge_table['RF'] = interest_rate 



	# return_vec = (np.asarray(merge_table).T)

	# pbar = opt.matrix(np.mean(return_vec, axis = 1))
	# pbar = pbar + mean_rollover
	# print pbar 


	n_portfolios = 5000
	means, stds = np.column_stack([random_portfolio(return_vec) for _ in xrange(n_portfolios)])

	#Compute minimum variance portfolio to match desired expected return, print optimal portfolio weights
	solvers.options['show_progress'] = False
	weights, returns_rmin, risks_rmin, expected_return, expected_std= OptimalWeights(return_vec, rminimum, pbar)
	risks, returns = EfficientFrontier(return_vec, pbar)
	print weights
	print 'Expected Vol'
	print expected_std
	print 'Expected Ret'
	print expected_return
	print 'RMin Returns'
	print returns_rmin


	plt.plot(stds, means, 'o')
	plt.plot(risks, returns, 'y-o')
	plt.plot(expected_std, expected_return, 'r*', ms= 16)
	plt.ylabel('Expected Return')
	plt.xlabel('Expected Volatility')
	plt.title('Portfolio Efficient Frontier')
	plt.show()


	# return weights, returns_rmin, risks_rmin

# def get_currency_list():
# 	currency_list = ['CURRFX/MXNUSD.1', 'CURRFX/USDCAD.1', 'CURRFX/NZDUSD.1', 'CURRFX/USDHKD.1', 'CURRFX/USDJPY.1', 'CURRFX/USDSGD.1', 'CURRFX/GBPUSD.1', 'CURRFX/USDZAR.1', 'CURRFX/AUDUSD.1', 'CURRFX/EURUSD.1']
# 	return currency_list

def merge_tables(returns_table, rollover_table):
	merge_table = pd.DataFrame(columns=rollover_table.columns)
	for index, column in enumerate(returns_table):
		merge_table[merge_table.columns[(index * 2)]] = -1 * returns_table.ix[:,index]
		merge_table[merge_table.columns[(index * 2) + 1]] = returns_table.ix[:,index]
	#When rollover table is large, the following will no longer take place.  For now, rollover must be calculated as an average rollover vector.
	# merge_table = merge_table + rollover_table
	return merge_table
  
# def get_currency_data(currency_list, num_days, shift, api_key):
# 	# Calculate dates
# 	end_date = datetime.date.today()
# 	start_date = end_date - timedelta(num_days)

# 	# Initialize data table
# 	data_table = None
# 	# Run through currencies, first assignment is initialized
# 	# Anything past first currency is joined into table
# 	for currency in currency_list:
# 		current_column = qdl.get(currency, start_date= start_date, end_date= end_date, authtoken= api_key)
# 		if data_table is None:
# 			data_table = current_column
# 		else:
# 			data_table = data_table.join(current_column, how= 'left', rsuffix= ' ')

# 	data_table.columns = ['MXN/USD', 'USD/CAD', 'NZD/USD', 'USD/HKD', 'USD/JPY', 'USD/SGD', 'GBP/USD', 'USD/ZAR', 'AUD/USD', 'EUR/USD']

# 	# Convert price data to returns and delete NaNs
# 	returns_table = data_table.pct_change(periods= shift).dropna()
# 	# Specify number of days to shift

# 	returns_table.drop(returns_table.index[:1], inplace=True)

# 	return returns_table

def rand_weights(n):
	''' Produces n random weights that sum to 1 '''
	k = np.random.rand(n)
	return k / sum(k)


def random_portfolio(returns_table):

	#Returns the mean and standard deviation of returns for a random portfolio

	p = np.asmatrix(np.mean(returns_table, axis=1))
	w = np.asmatrix(rand_weights(returns_table.shape[0]))
	C = np.asmatrix(np.cov(returns_table))
	
	mu = w * p.T
	sigma = np.sqrt(w * C * w.T)
	return mu, sigma

def OptimalWeights(returns, rmin, pbar):
	n = len(returns)
	returns = np.asmatrix(returns)
	# Convert to cvxopt matrices
	S = opt.matrix(np.cov(returns))
	print 'rmin value passed through as parameter'
	print rmin 
	N=2
	mus_min=max(min(pbar), 0)
	mus_max=max(pbar)
	mus_step=(mus_max - mus_min) / (N-1)
	mus = [mus_min + i*mus_step for i in range(N)]
	
	G = opt.matrix(np.concatenate((-np.transpose(pbar),-np.identity(n)),0))
	print G 
	A = opt.matrix(1.0, (1, n))
	b = opt.matrix(1.0)

	# Calculate efficient frontier weights using quadratic programming

	h=opt.matrix(np.concatenate((-np.ones((1,1))*rmin, np.zeros((n,1))),0))
	print h 
	portfolios = [solvers.qp(S, -pbar, G, h, A, b)['x'] for mu in mus]



	## CALCULATE RISKS AND RETURNS FOR FRONTIER
	returns_rmin = [blas.dot(pbar, x) for x in portfolios]
	risks_rmin = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
	## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
	m1 = np.polyfit(returns_rmin, risks_rmin, 2)
	x1 = np.sqrt(m1[2] / m1[0])
	
	# CALCULATE THE OPTIMAL PORTFOLIO
	h_mod= opt.matrix(np.concatenate((-np.ones((1,1))*x1, np.zeros((n,1))),0))
	wt = solvers.qp(S, -pbar, G, h_mod, A, b)
	sol = opt.matrix(wt['x'])

	expected_return = [blas.dot(pbar, sol)]
	expected_std = np.sqrt(sum([blas.dot(sol, S*sol)]))

	return np.asarray(sol), returns_rmin, risks_rmin, expected_return, expected_std

def EfficientFrontier(returns, pbar):

	n = len(returns)
	returns = np.asmatrix(returns)

	# Convert to cvxopt matrices
	S = opt.matrix(np.cov(returns))
	
	N=25
	mus_min=max(min(pbar),0)
	mus_max=max(pbar)
	mus_step=(mus_max - mus_min) / (N-1)
	mus = [mus_min + i*mus_step for i in range(N)]
	
	G = opt.matrix(np.concatenate((-np.transpose(pbar),-np.identity(n)),0))
	A = opt.matrix(1.0, (1, n))
	b = opt.matrix(1.0)

	# Calculate efficient frontier weights using quadratic programming
	portfolios=[]
	for r_min in mus:
		h=opt.matrix(np.concatenate((-np.ones((1,1))*r_min,np.zeros((n,1))),0))
		sol = solvers.qp(S, -pbar, G, h, A, b)['x']
	   
		portfolios.append(sol)

	## CALCULATE RISKS AND RETURNS FOR FRONTIER
	returns = [blas.dot(pbar, x) for x in portfolios]
	risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]

	return risks, returns

if __name__ == "__main__":
	main()