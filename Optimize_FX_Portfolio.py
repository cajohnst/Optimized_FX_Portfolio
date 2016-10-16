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
import Set_Variables as sv 

def main():
	# Authorization key for quandl
	auth_tok = sv.auth_tok
	# The last day to import data (most recent is today)
	end_date = sv.end_date
	np.random.seed(919)
	# Get currency lists from Pull_Data
	currency_list = Pull_Data.get_currency_list()
	currency_quandl_list = Pull_Data.get_currency_quandl_list()
	# Input number of days to retrieve historical rates
	num_days = sv.num_days_optimal_portfolio 
	#Compute returns with shift percentage change delay (daily = 1)
	shift = sv.shift
	# Input Leverage
	leverage = sv.leverage
	#For simplicity, assume fixed interest rate (risk free rate)
	interest_rate = sv.interest_rate
	# Minimum desired return
	rmin = sv.rmin

	#Compute returns
	currency_table = Pull_Data.get_currency_data(currency_list, currency_quandl_list, num_days, end_date, auth_tok)
	returns_table = currency_table.pct_change(periods= shift).dropna()
	returns_table.drop(returns_table.index[:1], inplace=True)

	rollover_table = rollover_google_sheet.pull_data(num_days)
	# Rollover from Forex.com is given in $10,000 increments, therefore daily return (as percentage) is rollover/(10,000 * 100) 
	rollover_table = rollover_table / 100

	# For the time-being, there is insufficient rollover data, we will calculate rollover as an average until this is no longer the case
	mean_rollover = np.mean(rollover_table, axis=0)
	mean_rollover = leverage * opt.matrix(np.append(mean_rollover, np.array(0)))
	# Merge returns and rollover tables
	merge_table = merge_tables(returns_table, rollover_table)
	merge_table = 100 * leverage * merge_table.dropna()
	merge_table['RF'] = interest_rate 

	# Convert returns to a numpy array to be used by cvxopt
	return_vec = (np.asarray(merge_table).T) 
	# Calculate the mean values of return_vec
	pbar = opt.matrix(np.mean(return_vec, axis = 1))
	# Add mean returns from currency pairs as well as rollovers
	pbar = pbar + mean_rollover
	# Input number of random portfolios to calculate
	n_portfolios = sv.n_portfolios
	# Calculate expected returns and standard deviation tuples for randomly weighted portfolios
	means, stds = np.column_stack([random_portfolio(return_vec) for _ in xrange(n_portfolios)])

	#Compute minimum variance portfolio to match desired expected return, print optimal portfolio weights
	solvers.options['show_progress'] = False
	weights, expected_return, expected_std= OptimalWeights(return_vec, rmin, pbar)
	risks, returns = EfficientFrontier(return_vec, pbar)

	return weights, expected_return, expected_std, risks, returns, means, stds


def merge_tables(returns_table, rollover_table):
	#Merge returns table and rollover tables, this is merely for formatting until rollover table is large.
	merge_table = pd.DataFrame(columns=rollover_table.columns)
	for index, column in enumerate(returns_table):
		merge_table[merge_table.columns[(index * 2)]] = -1 * returns_table.ix[:,index]
		merge_table[merge_table.columns[(index * 2) + 1]] = returns_table.ix[:,index]
	#When rollover table is large, the following will no longer take place.  For now, rollover must be calculated as an average rollover vector.
	# merge_table = merge_table + rollover_table
	return merge_table


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
	# Input number of iterations for mu (the number of points forming the efficient frontier)
	N=2
	# Iterations of mu are based on historical data
	mus_min=max(min(pbar), 0)
	mus_max=max(pbar)
	mus_step=(mus_max - mus_min) / (N-1)
	mus = [mus_min + i*mus_step for i in range(N)]
	# The following is an alternative way of calculating mus
	# mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
	
	# cvxopt parameters: Gx <= h, Ax <= b
	G = opt.matrix(np.concatenate((-np.transpose(pbar),-np.identity(n)),0)) 
	h=opt.matrix(np.concatenate((-np.ones((1,1))*rmin, np.zeros((n,1))),0))
	A = opt.matrix(1.0, (1, n))
	b = opt.matrix(1.0)

	# Calculate efficient frontier weights using quadratic programming
	portfolios = [solvers.qp(S, -pbar, G, h, A, b)['x'] for mu in mus]

	# Calculate risks and returns for the minimum return portfolio
	returns_rmin = [blas.dot(pbar, x) for x in portfolios]
	risks_rmin = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
	# Calculate the 2nd degree polynomial of the frontier curve at the minimum return portfolio
	m1 = np.polyfit(returns_rmin, risks_rmin, 2)
	x1 = np.sqrt(m1[2] / m1[0])
	
	# Calculate the Optimal Portfolio
	h_mod= opt.matrix(np.concatenate((-np.ones((1,1))*x1, np.zeros((n,1))),0))
	wt = solvers.qp(S, -pbar, G, h_mod, A, b)
	sol = opt.matrix(wt['x'])

	expected_return = [blas.dot(pbar, sol)]
	expected_std = np.sqrt(sum([blas.dot(sol, S*sol)]))
	print 'Weights'
	print list(sol)
	print 'Expected Vol'
	print expected_std
	print 'Expected Ret'
	print expected_return

	return list(sol), expected_return, expected_std


def EfficientFrontier(returns, pbar):

	n = len(returns)
	returns = np.asmatrix(returns)

	# Convert to cvxopt matrices
	S = opt.matrix(np.cov(returns))
	# Number of points to form efficient frontier (A greater N value requries more iterations and thus 
	# assumes a more accurate representation of the true curve)
	N=25
	mus_min=max(min(pbar),0)
	mus_max=max(pbar)
	mus_step=(mus_max - mus_min) / (N-1)
	mus = [mus_min + i*mus_step for i in range(N)]
	# cvxopt constraints: Gx <= h, Ax <= b
	G = opt.matrix(np.concatenate((-np.transpose(pbar),-np.identity(n)),0))
	A = opt.matrix(1.0, (1, n))
	b = opt.matrix(1.0)

	# Calculate efficient frontier weights using quadratic programming
	portfolios=[]
	for r_min in mus:
		h=opt.matrix(np.concatenate((-np.ones((1,1))*r_min,np.zeros((n,1))),0))
		sol = solvers.qp(S, -pbar, G, h, A, b)['x']
	   
		portfolios.append(sol)

	# Calculate risks and returns for the frontier
	returns = [blas.dot(pbar, x) for x in portfolios]
	risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]

	return risks, returns

if __name__ == "__main__":
	main()
	