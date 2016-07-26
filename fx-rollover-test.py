import pandas
from pandas.io.data import DataReader
import datetime
from datetime import date, timedelta

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers

np.random.seed(123)

# Turn off progress printing 
solvers.options['show_progress'] = False

symbol_dict = {'US':'USD','EU':'EUR', 'UK':'GBP', 'AL':'AUD', 'NZ':'NZD', 'JP':'JPY', 'MX':'MXN', 'SI':'SGD', 'SF':'ZAR', 'HK':'HKD'}

currency_list = ['DEXUSEU', 'DEXUSUK', 'DEXUSAL', 'DEXSFUS', 'DEXUSNZ', 'DEXSIUS', 'DEXMXUS', 'DEXJPUS', 'DEXCAUS', 'DEXHKUS']

num_days= 1000

def main():
    data_table = get_currency_data(currency_list)
    returns_table = get_returns(data_table)
    return returns_table

def get_currency_data(currency_list):
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

    # Drop NaNs
    data_table = data_table.dropna()

    return data_table.dropna()

def get_returns(data_table):
    # Calculate percentage changed
    returns_table = data_table.pct_change()
    # Remove first row for NaNs
    returns_table.drop(returns_table.index[:1], inplace=True)
    return returns_table
    
""" Optimization Step
Maximize interest rate return while minimizing variance in day to day moves.  This does not take moves in exchange rate into account currently.
Rollover table: Columns: Pairs, Rows: Short/Long
Eventually, will take average of 60-100 days of rollovers
Rollover_table: Divide each by 10,000
Pair    EURUSD    AUDUSD    GBPUSD    NZDUSD    USDMXN    USDJPY    USDCAD    USDHKD    USDZAR      USDSGD 
    S    0.96     -1.12      -.02      -1.32     2.79     -1.03      -.11      -.26     5.74        -.41
    L    -1.06    0.88       .26       1.04     -3.17      0.92      -.05      .10      -6.13        .33
    
#The following is borrowed from this link: http://blog.quantopian.com/markowitz-portfolio-optimization-2/
"""

# ## NUMBER OF ASSETS
# n_assets = len(rollover_table.columns)
# n_obs= num_days


# return_vec = np.random.randn(n_assets, n_obs)

# def optimal_portfolio(rollover_table):
#     n = len(rollover_table)
#     returns = np.asmatrix(rollover_table)
    
#     N = 100
#     mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
#     # Convert to cvxopt matrices
#     S = opt.matrix(np.cov(returns))
#     pbar = opt.matrix(np.mean(returns, axis=1))
    
#     # Create constraint matrices
#     G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
#     h = opt.matrix(0.0, (n ,1))
#     A = opt.matrix(1.0, (1, n))
#     b = opt.matrix(1.0)
    
#     # Calculate efficient frontier weights using quadratic programming
#     portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
#                   for mu in mus]
#     ## CALCULATE RISKS AND RETURNS FOR FRONTIER
#     returns = [blas.dot(pbar, x) for x in portfolios]
#     risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
#     ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
#     m1 = np.polyfit(returns, risks, 2)
#     x1 = np.sqrt(m1[2] / m1[0])
#     # CALCULATE THE OPTIMAL PORTFOLIO
#     wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
#     return np.asarray(wt), returns, risks
#     weights, returns, risks = optimal_portfolio(return_vec)

# plt.plot(stds, means, 'o')
# plt.ylabel('mean')
# plt.xlabel('std')
# plt.plot(risks, returns, 'y-o')
    
if __name__ == "__main__":
    foo = main()