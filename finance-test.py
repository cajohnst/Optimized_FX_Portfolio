import pandas
from pandas.io.data import DataReader
import datetime
from datetime import date, timedelta

symbol_dict = {'US':'USD','EU':'EUR', 'UK':'GBP', 'AL':'AUD', 'NZ':'NZD', 'JP':'JPY', 'MX':'MXN', 'SI':'SGD', 'SF':'ZAR', 'HK':'HKD'}

currency_list = ['DEXUSEU', 'DEXUSUK', 'DEXUSAL', 'DEXSFUS', 'DEXUSNZ', 'DEXSIUS', 'DEXMXUS', 'DEXJPUS', 'DEXCAUS', 'DEXHKUS']

def main():
    data_table = get_currency_data(currency_list)
    returns_table = get_returns(data_table)
    correlations_table = get_correlations(returns_table)
    return correlations_table
    print correlations_table

def get_currency_data(currency_list):
    # Calculate dates
    today = datetime.date.today()
    six_months_ago = today - timedelta(days=180)

    # Initialize data table
    data_table = None
    # Run through currencies, first assignment is initialized
    # Anything past first currency is joined into table
    for currency in currency_list:
        current_column = DataReader(currency, 'fred', six_months_ago, today)
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

def get_correlations(returns_table):
    # Calculate correlations
    return returns_table.corr()
    
if __name__ == "__main__":
    foo = main()


""" Optimization Step
Maximize interest rate return while minimizing variance in day to day moves.  This does not take moves in exchange rate into account currently.
Rollover table: Columns: Pairs, Rows: Short/Long
Eventually, will take average of 60-100 days of rollovers
Rollover_table:
Pair    EURUSD    AUDUSD    GBPUSD    NZDUSD    USDMXN    USDJPY    USDCAD    USDHKD    USDZAR      USDSGD 
    S    0.96     -1.12      -.02      -1.32     2.79     -1.03      -.11      -.26     5.74        -.41
    L    -1.06    0.88       .26       1.04     -3.17      0.92      -.05      .10      -6.13        .33
    
#https://wellecks.wordpress.com/2014/03/23/portfolio-optimization-with-python/

#Borrowed the following from link above, I think it should work just need to implement rollover_table
import cvxopt 
average_return= matrix(rollover_table)
sigma= matrix(covariance_table)
min_return= 0.05
#Since portfolio maximization is quadratic, let P= sigma
P= sigma
q= matrix(numpy.zeros((n, 1)))

#subject to constraints average return >= value and weights are all greater than 0 and sum to 1

G = matrix(numpy.concatenate((
             -numpy.transpose(numpy.array(average_return)), 
             -numpy.identity(n)), 0))
h = matrix(numpy.concatenate((
             -numpy.ones((1,1))*min_return, 
              numpy.zeros((n,1))), 0))
              
A = matrix(1.0, (1,n))
b = matrix(1.0)

sol = solvers.qp(P, q, G, h, A, b)
"""