#Set variables
import datetime
from datetime import date, timedelta
# Define quandl list as list of quandl codes to be used in portfolio
# Define currency list as names of currencies available for portfolio
def get_currency_quandl_list():
	currency_quandl_list = ['CURRFX/MXNUSD.1', 'CURRFX/USDCAD.1', 'CURRFX/NZDUSD.1', 'CURRFX/USDHKD.1', 'CURRFX/USDJPY.1', 'CURRFX/USDSGD.1', 'CURRFX/GBPUSD.1', 'CURRFX/USDZAR.1',
							'CURRFX/AUDUSD.1', 'CURRFX/EURUSD.1']
	return currency_quandl_list

def get_currency_list():
	currency_list = ['USD/MXN', 'USD/CAD', 'NZD/USD', 'USD/HKD', 'USD/JPY', 'USD/SGD', 'GBP/USD', 'USD/ZAR', 'AUD/USD', 'EUR/USD']
	return currency_list 
# Execution Cost List as decimal (In the order of currency list)
execution_costs = [.008, 0.00027, 0.0003, 0.0012, 0.2, 0.00053, 0.00029, 0.002, 0.0002, 0.00017]
# authorization key for quandl data 
auth_tok = "kz_8e2T7QchJBQ8z_VSi"
# Input last day to get returns data for (default is today)
end_date = datetime.date.today() 
# Input original portfolio value, used for VaR calculations
portfolio_value = 1000
# Input number of days to calculate back returns
num_days_optimal_portfolio = 200
#Compute returns with shift percentage change delay (daily = 1)
shift = 1
# Input Leverage
leverage = 10
# If daily return has been met, reduce leverage to this factor
reduced_leverage = 2.5
# Input Rolling Period for moving averages
rolling_period = 2
# Input minimum desired return for portfolio optimization
rminimum = 100/float(252)
# Input risk free interest rate
interest_rate = 2/ float(365)
# Input weight threshold to make new trade 
weight_threshold = 0.03
# Input interval for displaying changes in the weight distribution over time for distribution chart (5 = 5 portfolio changes)
distribution_interval = 5
# Number of days to pull data for 
# ** this num_days is a different value than that used in other files **
num_days_regression = 720
# ** Before this date, daily high/low data is unreliable and therefore the Stochastic calculation is unreliable **
stoch_date = datetime.date(2016, 7, 15)
# Number of random portfolios in Optimize_FX_Portfolio
n_portfolios = 5000

#Number of days worth of data useable for charts or regression analysis
num_days_charts = 100
#q = avg. periods for gain/loss
q = 14
# On the scale from 0-100, this level is considered to be "overbought" by RSI, typical value is 70
Overbought = 70
#On the scale from 0-100, this level is considered to be "oversold" by RSI, typical value is 30
Oversold = 30
#Determine the moving average windows for MACD, moving average convergence divergence, as measured by
#the difference between slow and fast exponentially weighted moving averages compared to the fastest of 
#the three.  Levels are typically 26 for slow, 12 for fast, and 9 for fastest
nslow = 26
nfast = 12
nema = 9
#Determine windows for simple moving averages to be overlayed on the exchange rate chart.  Levels vary, but widely-used
#rolling averages include 10, 20, 50, 100, and 200 day averages
ma_slow = 100
ma_fast = 20
#Determine windows for stochastics.  A typical window is 14 periods.  N is the number of windows.  D is the "slow" stochastic window
#typically a 3- period moving average of the fast stochastic
n = 14
d = 3
# RSI overbought or oversold
Overbought_S = 80
Oversold_S = 20
# Number of bins for VaR histogram in Daily_Reports
num_bins = 10