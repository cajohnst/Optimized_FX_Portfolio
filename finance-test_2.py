import pandas as pd
from pandas.io.data import DataReader
import datetime
from datetime import date, timedelta

import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers

import rollover_scraper

np.random.seed(1)

# Turn off progress printing 
solvers.options['show_progress'] = False

currency_list = ['DEXUSEU', 'DEXUSUK', 'DEXUSAL', 'DEXSFUS', 'DEXUSNZ', 'DEXSIUS', 'DEXMXUS', 'DEXJPUS', 'DEXCAUS', 'DEXHKUS']

def main():
    num_days = 365
    # data_table = get_currency_data(currency_list, num_days)
    returns_table = get_currency_data(currency_list, num_days)

    #rollover_table = rollover_scraper.generate_rollover(currency_list)

    random_weights= rand_weights(len(returns_table.columns))
    print rand_weights(len(returns_table.columns))
    
    n_portfolios = 5000
    
    means, stds = np.column_stack([random_portfolio(returns_table) for _ in xrange(n_portfolios)])
    
    plot_1 = plt.figure(1)

    plt.plot(stds, means, 'o', markersize=5)
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.title('Mean and standard deviation of returns of randomly generated portfolios')

    plot_1.show()

    # N is the number of points generated on the efficient frontier curve
    N = 500

    weights, returns, risks = optimal_portfolio(returns_table.T, N)

    plot_2 = plt.figure(2)
    plt.plot(stds, means, 'o')
    plt.ylabel('mean')
    plt.xlabel('std')
    plt.plot(risks, returns, 'y-o')

    plot_2.show()

    print 'Efficient Weights'
    print weights

    input()


def get_currency_data(currency_list, num_days):
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
    returns_table = data_table.pct_change().dropna()
    returns_table.drop(returns_table.index[:1], inplace=True)
    return returns_table

# def get_returns(data_table):
#     # Calculate percentage changed
#     returns_table = data_table.pct_change()
#     # Remove first row for NaNs
#     returns_table.drop(returns_table.index[:1], inplace=True)
#     return returns_table
    
""" Optimization Step
Maximize interest rate return while minimizing variance in day to day moves.  This does not take moves in exchange rate into account currently.
Rollover table: Columns: Pairs, Rows: Short/Long
Eventually, will take average of 60-100 days of rollovers
    
#The following is borrowed from this link: http://blog.quantopian.com/markowitz-portfolio-optimization-2/
"""

def rand_weights(n):
#For now, want random weights ranging from 0 to 1 in 0.1 intervals
    # k = np.random.randint(0, 10, size=(10,1)).astype("float") / 10
    k = np.random.rand(n)
    return k / sum(k)

def random_portfolio(returns):
    # Returns the mean and standard deviation of returns for a random portfolio

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma


def optimal_portfolio(returns, N):
    n = len(returns)
    returns = np.asmatrix(returns)
    # N = number of points on efficient frontier
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    # S = covariance matrix, pbar = mean of return data set
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))

    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    print 'S'
    print S
    print 'mus'
    print mus
    print 'pbar'
    print pbar
    print 'G'
    print G
    print 'h'
    print h
    print 'A'
    print A
    print 'b'
    print b
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']

    return np.asarray(wt), returns, risks


if __name__ == "__main__":
 weights, returns, risks, returns_table = main()


    

