import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import datetime
from datetime import date, timedelta
import quandl as qdl



''' 1. Pull_data
	2. Run RSI_sample, MACD_sample, Futures_vs_Spot, and Events
	3. Run regression models for different event scenarios along with the rest of the info
	4. Use regression model to adjust pbar
	5. Run Optimize_FX_Portfolio with adjusted pbar
	6. Produce_charts showing regression models for each currency in currency list, models for days events, efficient frontier,
		and finally portfolio statistics (show portfolio distribution, charts for each holding (with RSI and MACD), and chart of portfolio vs. various metrics)
	7. Save as a markdown or PDF
	8. Automate run to Heroku in early am

	'''
def main():
	auth_tok = "kz_8e2T7QchJBQ8z_VSi"

	currency_dictionary = currency_dict()
	fed_dictionary = fed_dict()

	#Number of days worth of data useable for charts or regression analysis
	num_days = 100
	#The last (most recent) data point to pull for data tables
	to_date = datetime.date.today()
	#Create new lists to pull daily lows and highs for the stochastic oscillator
	dict_high = [high.replace('1', '2') for high in currency_dictionary]
	dict_low = [low.replace('1', '3') for low in currency_dictionary]

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

	Overbought_S = 80
	Oversold_S = 20

	max_lag = max(q, nslow, nfast, nema, ma_slow, ma_fast, n, d)

	#Pull this many days of data to return the amount of data user has requested
	pull_data_days = num_days + max_lag

	#Pull data from quandl
	currency_table = get_currency_data(currency_dictionary, pull_data_days, to_date, auth_tok)
	#Get daily lows from quandl for stochastic oscillator
	low_table = get_currency_data(dict_low, pull_data_days, to_date, auth_tok)
	#Get daily highs from quandl for stochastic oscillator
	high_table = get_currency_data(dict_high, pull_data_days, to_date, auth_tok)

	# #Calculate RSI for all currency pairs in currency_table
	RSI = RSI_Calc(currency_table, q)

	# #Calculate simple moving averages
	ma_f = moving_average(currency_table, ma_fast, type='simple')
	ma_s = moving_average(currency_table, ma_slow, type='simple')

	#Calculate exponentially weighted moving averages and MACD
	emaslow, emafast, macd = get_MACD(currency_table, nslow= nslow, nfast = nfast)
	ema9 = moving_average(macd, nema, type = 'exponential')

	#Calculate stochastics
	fast_stochastic, slow_stochastic = get_stochastic(currency_table, low_table, high_table, n, d)

	RSI = drop_rows(RSI, max_lag)
	ma_f = drop_rows(ma_f, max_lag)
	ma_s = drop_rows(ma_s, max_lag)
	emaslow= drop_rows(emaslow, max_lag)
	emafast= drop_rows(emafast, max_lag)
	macd = drop_rows(macd, max_lag)
	ema9 = drop_rows(ema9, max_lag)
	fast_stochastic = drop_rows(fast_stochastic, max_lag)
	slow_stochastic = drop_rows(slow_stochastic, max_lag)
	currency_table = drop_rows(currency_table, max_lag)

	# Convert price data to returns and delete NaNs
	shift = 1
	returns_table = currency_table.pct_change(periods= shift).dropna()
	# Specify number of days to shift

	returns_table.drop(returns_table.index[:1], inplace=True)

	fed_table = get_fed_data(fed_dictionary, pull_data_days, to_date, auth_tok)
	fed_table = drop_rows(fed_table, max_lag)

	return returns_table, RSI, ma_f, ma_s, macd, ema9, fast_stochastic, currency_table, fed_table
#################################################################################################################################
# Pull Data
def currency_dict():
	currency_dict = {'MXN/USD':'CURRFX/MXNUSD.1', 'USD/CAD':'CURRFX/USDCAD.1', 'NZD/USD':'CURRFX/NZDUSD.1', 'USD/HKD':'CURRFX/USDHKD.1', 
						'USD/JPY':'CURRFX/USDJPY.1', 'USD/SGD':'CURRFX/USDSGD.1', 'GBP/USD':'CURRFX/GBPUSD.1', 'USD/ZAR':'CURRFX/USDZAR.1',
						 'AUD/USD':'CURRFX/AUDUSD.1', 'EUR/USD':'CURRFX/EURUSD.1'}
	return currency_dict

def get_currency_data(currency_dict, num_days, end_date, api_key):
	# Calculate dates
	start_date = end_date - timedelta(num_days)

	# Initialize data table
	data_table = None
	# Run through currencies, first assignment is initialized
	# Anything past first currency is joined into table
	for currency, quandl_name in currency_dict.items():
		current_column = qdl.get(quandl_name, start_date= start_date, end_date= end_date, authtoken= api_key)
		current_column.columns = [currency]
		if data_table is None:
			data_table = current_column
		else:
			data_table = data_table.join(current_column, how= 'left', rsuffix= ' ')

	return data_table

''' This section will also contain pull data for econ dict when it is completed '''

# def us_economic_dict():
# 	us_econ_dict = {'Consumer Price Index (YoY)':'fred/cpiaucsl', 'Consumer Price Index Ex Food & Energy (YoY)': 'fred/cpilfesl', 'Nonfarm Payrolls': 'fred/payems', 'Reuters/Michigan Consumer Sentiment Index': 'umich/soc1', 
# 					'Baker Hughes US Oil Rig Count': 'bkrhughes/rigs_by_state_totalus_total', 'PMI Composite': 'fred/napm', 'Durable Goods': 'fred/dgorder', 
# 					'Retail Sales (MoM)': 'fred/rsafs', 'Initial Jobless Claims': 'fred/icsa', 'ADP': 'adp/empl_sec', 'GDP': 'fred/gdp', 
# 					'Unemployment Rate': 'fred/unrate', 'M2': 'fred/m2', 'Housing Starts': 'fred/houst', '10-Year Yield': 'fred/dgs10'}

# 	return us_econ_dict

# def euro_dict():
# 	euro_econ_dict = {}
# 	return euro_econ_dict

# def jpy_dict():
# 	jpy_econ_dict = {}
# 	return jpy_econ_dict

# def gbp_dict():
# 	gbp_econ_dict = {}
# 	return gbp_econ_dict

# def aud_dict():
# 	aud_econ_dict = {}
# 	return aud_econ_dict

# def cad_dict():
# 	cad_econ_dict = {}
# 	return cad_econ_dict 

# def cny_dict():
# 	cny_econ_dict = {}
# 	return cny_econ_dict

# def nzd_dict():
# 	nzd_econ_dict = {}
# 	return nzd_econ_dict

def fed_dict():
	fed_dict = {'Federal Funds Futures': 'CHRIS/CME_FF1', 'Effective Funds Rate': 'FRED/DFF'}
	return fed_dict

def get_fed_data(fed_dictionary, num_days, end_date, api_key):
	start_date = end_date - timedelta(num_days)
	# Initialize data table
	fed_table = None
	# Run through currencies, first assignment is initialized
	# Anything past first currency is joined into table
	for fed_data, quandl_name in fed_dictionary.items():
		current_column = qdl.get(quandl_name, start_date= start_date, end_date= end_date, authtoken= api_key)
		current_column.columns = [fed_data]
		if fed_table is None:
			fed_table = current_column
		else:
			fed_table = fed_table.join(current_column, how= 'left', rsuffix= ' ')
	fed_table['rate_hike_prob_25_basis'] = (fed_table['Federal Funds Futures'] - fed_table['Effective Funds Rate'])/ 0.25
	fed_table['rate_hike_prob_50_basis'] = (fed_table['Federal Funds Futures'] - fed_table['Effective Funds Rate'])/ 0.5
	fed_table = fed_table.drop('Federal Funds Futures')
	fed_table = fed_table.drop('Effective Funds Rate')

	return fed_table

################################################################################################################################
# Calculate Technicals

def RSI_Calc(currency_data, q):
	delta = currency_data.diff()
	dUp, dDown = delta.copy(), delta.copy()
	dUp[dUp < 0] = 0
	dDown[dDown > 0] = 0

	RolUp = dUp.rolling(window= q, center= False).mean()
	RolDown = dDown.rolling(window = q, center= False).mean().abs()

	RS = RolUp / RolDown
	RSI = 100.0 - (100.0 / (1.0 + RS))
	return RSI

def moving_average(x, n, type='simple'):
	""" compute an n period moving average.
		type is 'simple' | 'exponential'
	"""
	if type == 'simple':
		ma = x.rolling(window = n, center= False).mean()

	else:
		ma = x.ewm(span = n).mean()
	return ma

def get_MACD(x, nslow=26, nfast=12):
	"""
	compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
	return value is emaslow, emafast, macd which are len(x) arrays
	"""
	emaslow = moving_average(x, nslow, type='exponential')
	emafast = moving_average(x, nfast, type='exponential')
	macd = emafast - emaslow

	return emaslow, emafast, macd

def get_stochastic(data, data_low, data_high, n, smoothing):
	low = data_low.rolling(window = n, center= False).min()
	high = data_high.rolling(window= n, center= False).max()
	k = 100 * ((data - low)/ (high - low))
	d = moving_average(k, smoothing, type = 'simple')

	return k, d

def drop_rows(data, max_val):
	data = data.ix[max_val:]

	return data

def merge_technicals(currency_dict, RSI, MACD, Stochastic):
	for currency in currency_dict.items():
		merge_tech_table[currency] = RSI[currency]
		merge_tech_table[currency] = merge_tech_table.join(MACD[currency], how= 'left', rsuffix= '')
		merge_tech_table[currency] = merge_tech_table.join(Stochastics[currency], how= 'left', rsuffix= '')
		merge_tech_table[currency] = merge_tech_table.join(fed_data, how= 'left', rsuffix= '')
		merge_tech_table.columns = ['RSI', 'MACD', 'Stochastic']



if __name__ == "__main__":
	main()










