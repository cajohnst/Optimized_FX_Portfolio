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
# from sklearn.covariance import OAS

def main():
	auth_tok = "kz_8e2T7QchJBQ8z_VSi"
	np.random.seed(919)
	currency_list = get_currency_list()
	num_days = 500
	#Compute returns with shift delay
	shift = 1
	#Compute returns
	# returns_table = get_currency_data(currency_list, num_days, shift, auth_tok)
	# return_vec = np.asarray(returns_table)
	# print return_vec

	n_assets = 10

	## NUMBER OF OBSERVATIONS
	n_obs = 5000

	return_vec = np.random.randn(n_assets, n_obs)

	#For simplicity, assume fixed interest rate
	interest_rate = 0.005/252.

	# Minimum desired return
	rmin = 0.225/252

	# rollover_table = rollover_google_sheet.pull_data(num_days)

	# rollover_table = rollover_table / 10000

	# merge_table = merge_tables(returns_table, rollover_table)
	# merge_table = merge_table.dropna()

	# # Input Leverage
	# leverage = 10/1
	# # Return the returns table with leverage
	# merge_table = merge_table * leverage

	# merge_table['RF'] = interest_rate

	n_portfolios = 5000
	means, stds = np.column_stack([random_portfolio(return_vec) for _ in xrange(n_portfolios)])

	#Compute minimum variance portfolio to match desired expected return, print optimal portfolio weights
	solvers.options['show_progress'] = False
	weights, returns_rmin, risks_rmin, expected_return, expected_std= OptimalWeights(return_vec, rmin)
	risks, returns = EfficientFrontier(return_vec)

	plt.plot(stds, means, 'o')
	plt.plot(risks, returns, 'y-o')
	plt.plot(expected_std, expected_return, 'r*', ms= 16)
	plt.ylabel('Expected Return')
	plt.xlabel('Expected Volatility')
	plt.title('Portfolio Efficient Frontier')
	plt.show()
	return weights, returns_rmin, risks_rmin

def get_currency_list():
	currency_list = ['CURRFX/MXNUSD.1', 'CURRFX/USDCAD.1', 'CURRFX/NZDUSD.1', 'CURRFX/USDHKD.1', 'CURRFX/USDJPY.1', 'CURRFX/USDSGD.1', 'CURRFX/GBPUSD.1', 'CURRFX/USDZAR.1', 'CURRFX/AUDUSD.1', 'CURRFX/EURUSD.1']
	return currency_list

def merge_tables(returns_table, rollover_table):
	merge_table = pd.DataFrame(columns=rollover_table.columns)
	for index, column in enumerate(returns_table):
		merge_table[merge_table.columns[(index * 2)]] = -1 * returns_table.ix[:,index]
		merge_table[merge_table.columns[(index * 2) + 1]] = returns_table.ix[:,index]
	merge_table = merge_table + rollover_table
	return merge_table
  
def get_currency_data(currency_list, num_days, shift, api_key):
	# Calculate dates
	end_date = datetime.date.today()
	start_date = end_date - timedelta(num_days)

	# Initialize data table
	data_table = None
	# Run through currencies, first assignment is initialized
	# Anything past first currency is joined into table
	for currency in currency_list:
		current_column = qdl.get(currency, start_date= start_date, end_date= end_date, authtoken= api_key)
		if data_table is None:
			data_table = current_column
		else:
			data_table = data_table.join(current_column, how= 'left', rsuffix= ' ')

	data_table.columns = ['MXN/USD', 'USD/CAD', 'NZD/USD', 'USD/HKD', 'USD/JPY', 'USD/SGD', 'GBP/USD', 'USD/ZAR', 'AUD/USD', 'EUR/USD']

	# Convert price data to returns and delete NaNs
	returns_table = data_table.pct_change(periods= shift).dropna()
	# Specify number of days to shift

	returns_table.drop(returns_table.index[:1], inplace=True)

	return returns_table

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


# def MarkowitzOpt(returns_table, returns_mean,returns_var,returns_cov,irate,rmin): 

def OptimalWeights(returns, rmin):
	n = len(returns)
	returns = np.asmatrix(returns)

	# Convert to cvxopt matrices
	S = opt.matrix(np.cov(returns))
	pbar = opt.matrix(np.mean(returns, axis=1))
	
	N=2
	mus_min=max(min(pbar),0)
	mus_max=max(pbar)
	mus_step=(mus_max - mus_min) / (N-1)
	mus = [mus_min + i*mus_step for i in range(N)]
	
	G = opt.matrix(np.concatenate((-np.transpose(pbar),-np.identity(n)),0))
	A = opt.matrix(1.0, (1, n))
	b = opt.matrix(1.0)

	# Calculate efficient frontier weights using quadratic programming
	h=opt.matrix(np.concatenate((-np.ones((1,1))*rmin,np.zeros((n,1))),0))
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

	print expected_return
	print expected_std

	return np.asarray(sol), returns_rmin, risks_rmin, expected_return, expected_std

def EfficientFrontier(returns):

	n = len(returns)
	returns = np.asmatrix(returns)

	# Convert to cvxopt matrices
	S = opt.matrix(np.cov(returns))
	pbar = opt.matrix(np.mean(returns, axis=1))
	
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