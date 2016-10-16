''' Pull all data to be used in Optimize_FX_Portfolio, RSI_sample, MACD_sample, and futures_vs_spot (eventually pull fred economic reports) '''
import pandas as pd 
import quandl as qdl 
import numpy as np 
from pandas import Series, DataFrame
import datetime
from datetime import timedelta, date
import rollover_google_sheet
import fxstreet_google_sheet
import RSI_sample
import fxstreet_scraper
import sklearn as sklearn
from sklearn import preprocessing
from sklearn.linear_model import Ridge
import StringIO
import csv 
import import_spot_test

def main():
	# Quandl Authorization key
	auth_tok = "kz_8e2T7QchJBQ8z_VSi"

	# Number of days to pull data for 
	# ** this num_days is a different value than that used in other files **
	num_days = 720
	# Leverage multiplier
	leverage = 10 
	# country_list = get_country_list()
	currency_list = get_currency_list()
	currency_quandl_list = get_currency_quandl_list()
	fed_list = get_fed_list()
	# Last day to retrieve data (most recent is today)
	end_date = datetime.date.today()
	# Calculate beginning date
	beg_date = end_date - timedelta(num_days)
	# ** Before this date, daily high/low data is unreliable and therefore the Stochastic calculation is unreliable **
	stoch_date = datetime.date(2016, 7, 15)


#############################################################################################################################
	# Import events data into pandas dataframe
	econ_calendar_full = pd.DataFrame(fxstreet_google_sheet.pull_data(num_days))
	# If no consensus value exists, use the previous value to fill its place.
	econ_calendar_full.Consensus.fillna(econ_calendar_full.Previous, inplace = True)
	# Create column 'Deviation' as the actual value on the data release subtracting the market expectation
	econ_calendar_full['Deviation'] = econ_calendar_full['Actual'] - econ_calendar_full['Consensus']

# #Take today's Economic Events from FXstreet_scraper and format
	econ_calendar_today = fxstreet_scraper.main()
	econ_calendar_today = StringIO.StringIO(econ_calendar_today)
	econ_calendar_today = pd.read_csv(econ_calendar_today, header=0, index_col= False)
	econ_calendar_today['DateTime'] = pd.to_datetime(econ_calendar_today['DateTime'])
	econ_calendar_today = econ_calendar_today.set_index('DateTime')
	econ_calendar_today.index.names = ['DateTime']
	econ_calendar_today.Consensus.fillna(econ_calendar_today.Previous, inplace = True)
	econ_calendar_today.dropna(thresh= 5, inplace = True)

	#Begin raw input of user predictions for data releases (fill 'Actual' column with predictions).
	for index, row in econ_calendar_today.iterrows():
		prediction = raw_input("Prediction for {0} in {1} given the market consensus is {2}.\n Your Prediction:".format(row['Name'], row['Country'], row['Consensus']))
		econ_calendar_today.set_value(index, 'Actual', prediction)
 	# Append today's data to the full calendar of releases
	econ_calendar_full = econ_calendar_full.append(econ_calendar_today)

#############################################################################################################################
#RSI_sample data
	#Determine windows for stochastics.  A typical window is 14 periods.  N is the number of windows.  D is the "slow" stochastic window
	#typically a 3- period moving average of the fast stochastic
	n = 14
	d = 3
	#Determine the moving average windows for MACD, moving average convergence divergence, as measured by
	#the difference between slow and fast exponentially weighted moving averages compared to the fastest of 
	#the three.  Levels are typically 26 for slow, 12 for fast, and 9 for fastest
	nslow = 26
	nfast = 12
	nema = 9
	#q = avg. periods for gain/loss
	q = 14
	#Determine windows for simple moving averages to be overlayed on the exchange rate chart.  Levels vary, but widely-used
	#rolling averages include 10, 20, 50, 100, and 200 day averages
	ma_slow = 100
	ma_fast = 20

	# For stochastic data, import daily highs and lows into dataframe from quandl
	# Highs and Lows are unreliable before stoch_date (and need to be continually monitored)
	list_high = [high.replace('1', '2') for high in currency_quandl_list]
	list_low = [low.replace('1', '3') for low in currency_quandl_list]
	# Maximum of the 'rolling' periods
	max_lag = max(q, nslow, nfast, nema, ma_slow, ma_fast, n, d)

	#Pull data from quandl
	currency_table = get_currency_data(currency_list, currency_quandl_list, num_days, end_date, auth_tok)

	live_rates = import_spot_test.main()
	currency_table = currency_table.append(live_rates)

	#Get daily lows from quandl for stochastic oscillator
	low_table = get_currency_data(currency_list, list_low, num_days, end_date, auth_tok)
	#Get daily highs from quandl for stochastic oscillator
	high_table = get_currency_data(currency_list, list_high, num_days, end_date, auth_tok)

	# #Calculate RSI for all currency pairs in currency_table
	RSI = RSI_sample.RSI_Calc(currency_table, q)

	#Calculate exponentially weighted moving averages and MACD
	emaslow, emafast, macd = RSI_sample.get_MACD(currency_table, nslow= nslow, nfast = nfast)
	ema9 = RSI_sample.moving_average(macd, nema, type = 'exponential')

	#Calculate stochastics
	fast_stochastic, slow_stochastic = RSI_sample.get_stochastic(currency_table, low_table, high_table, n, d)

	# #Calculate simple moving averages
	ma_f = RSI_sample.moving_average(currency_table, ma_fast, type='simple')
	ma_s = RSI_sample.moving_average(currency_table, ma_slow, type='simple')

	# Drop all NaNs, format data so indexes will match when joined
	RSI = RSI_sample.drop_rows(RSI, max_lag)
	ma_f = RSI_sample.drop_rows(ma_f, max_lag)
	ma_s = RSI_sample.drop_rows(ma_s, max_lag)
	emaslow= RSI_sample.drop_rows(emaslow, max_lag)
	emafast= RSI_sample.drop_rows(emafast, max_lag)
	macd = RSI_sample.drop_rows(macd, max_lag)
	ema9 = RSI_sample.drop_rows(ema9, max_lag)
	fast_stochastic = RSI_sample.drop_rows(fast_stochastic, max_lag)
	slow_stochastic = RSI_sample.drop_rows(slow_stochastic, max_lag)
	currency_table = RSI_sample.drop_rows(currency_table, max_lag)

#################################################################################################################
	#Create fundamentals, merge tables, perform ridge regression, output daily return predictions
	# Convert price data to returns and delete NaNs
	shift = 1
	returns_table = currency_table.pct_change(periods= shift).dropna()
	returns_table.drop(returns_table.index[:1], inplace=True)
	returns_table = 100 * leverage * returns_table 

	fed_table = get_fed_data(fed_list, num_days, end_date, auth_tok)

	# Specialize data for events! Pull all historical data from event calendar which matches name in econ data dictionary.
	economic_data_dict = get_economic_data_dict()
	all_fundamentals_table = query_past_economic_data(econ_calendar_today, econ_calendar_full, fed_table, economic_data_dict)
	#Merge the calendar data with the columns of technicals
	regression_table = merge_with_technicals(currency_list, returns_table, all_fundamentals_table, RSI, macd, fast_stochastic, beg_date, stoch_date)
	return regression_table 

def query_past_economic_data(calendar_today, calendar_full, fed_table, economic_data_dict):
	# Get all historical data from the full calendar which matches today's data and is found in the econ data dictionary.
	for index, values in calendar_today.iterrows():
		country = values['Country']
		if country in economic_data_dict:
			event_name = values['Name']
			if event_name in economic_data_dict[country]:
				pull_events = calendar_full[(calendar_full['Country'] == country) & (calendar_full['Name'] == event_name)]
				fed_table = fed_table.join(pull_events['Deviation'], how = 'left', rsuffix = ' of {0}'.format(event_name))
				fed_table = fed_table.fillna(value = 0)
			else:
				print ' *** {0} not a listed event for {1} in the Economic Dictionary ***'.format(event_name, country)
		else:
			print ' *** {0} is not a country in the Economic Dictionary ***'.format(country)

	return fed_table 

def get_currency_quandl_list():
	currency_quandl_list = ['CURRFX/MXNUSD.1', 'CURRFX/USDCAD.1', 'CURRFX/NZDUSD.1', 'CURRFX/USDHKD.1', 'CURRFX/USDJPY.1', 'CURRFX/USDSGD.1', 'CURRFX/GBPUSD.1', 'CURRFX/USDZAR.1',
							'CURRFX/AUDUSD.1', 'CURRFX/EURUSD.1']
	return currency_quandl_list

def get_currency_list():
	currency_list = ['USD/MXN', 'USD/CAD', 'NZD/USD', 'USD/HKD', 'USD/JPY', 'USD/SGD', 'GBP/USD', 'USD/ZAR', 'AUD/USD', 'EUR/USD']
	return currency_list 


def get_economic_data_dict():
	# Dictonary keys are the country name
	# Key values are tuples structured as ([list of eventnames from fxstreet], [list of eventnames from quandl])
	economic_data_dict = {
	'United States': 
		['Consumer Price Index (YoY)', 'Consumer Price Index Ex Food & Energy (YoY)', 'Nonfarm Payrolls', 'Reuters/Michigan Consumer Sentiment Index', 'Baker Hughes US Oil Rig Count', 'Durable Goods Orders', 'Durable Goods Orders ex Transportation', 'Retail Sales (MoM)', 'Initial Jobless Claims', 'ADP Employment Change', 'Gross Domestic Product Annualized', 'Unemployment Rate', 'M2', 'Housing Starts (MoM)', 'Building Permits (MoM)', '10-Year Note Auction', 
		'EIA Crude Oil Stocks change', 'S&P/Case-Shiller Home Price Indices (YoY)', 'Markit Services PMI', 'Markit PMI Composite', 'Consumer Confidence', 'Dallas Fed Manufacturing Business Index', 'ISM Prices Paid', 'ISM Manufacturing PMI', 'Markit Manufacturing PMI', 'Construction Spending (MoM)', 'Trade Balance', 'ISM Non-Manufacturing PMI', 'Factory Orders (MoM)']
	,
	'Japan':
		['National Consumer Price Index (YoY)', 'Foreign investment in Japan stocks', 'Foreign bond investment', 'Unemployment Rate', 'Industrial Production (MoM)', 'Industrial Production (YoY)' ]
	,
	'European Monetary Union':
		['Unemployment Rate', 'Consumer Price Index (YoY)', 'Consumer Price Index - Core (YoY)', 'Markit Manufacturing PMI', 'Producer Price Index (MoM)', 'Producer Price Index (YoY)', 'Markit Services PMI']
	,
	'Germany':
		['Markit Manufacturing PMI', '10-y Bond Auction']
	,
	'Australia':
		['TD Securities Inflation (YoY)', 'TD Securities Inflation (MoM)', 'RBA Interest Rate Decision', 'Retail Sales s.a (MoM)']
	,
	'Canada':
		['RBC Manufacturing PMI']
	,
	'New Zealand':
		[]
	,
	'China':
		[]
	,
	'United Kingdom':
		['Markit Manufacturing PMI', 'PMI Construction']
	, 
	'Italy':
		[]
	, 
	'Switzerland':
		['Real Retail Sales (YoY)']
	,
	'France':
		[]
	}

	return economic_data_dict 

def get_fed_list():
	fed_list = ['Federal_Funds_Futures', 'Effective_Funds_Rate']
	return fed_list 


def get_currency_data(currency_list, currency_quandl_list, num_days, end_date, api_key):
	# Calculate dates to begin and end
	start_date = end_date - timedelta(num_days)

	# Initialize data table
	data_table = None
	# Run through currencies, first assignment is initialized
	# Anything past first currency is joined into table 
	for currency in currency_quandl_list:
		current_column = qdl.get(currency, start_date= start_date, end_date= end_date, authtoken= api_key)
		current_column.columns = [currency]
		if data_table is None:
			data_table = current_column
		else:
			data_table = data_table.join(current_column, how= 'left', rsuffix= '')
	data_table.columns = currency_list 
	if 'USD/MXN' in currency_list:
		data_table['USD/MXN'] = 1 / data_table['USD/MXN']
	return data_table 

def get_fed_data(fed_reserve_list, num_days, end_date, api_key):
	# Calculate dates
	start_date = end_date - timedelta(num_days)
	# Get Federal Funds Futures data and Effective Funds Futures
	fed_fund_futures = qdl.get('CHRIS/CME_FF1.6', start_date= start_date, end_date= end_date, authtoken= api_key)
	effective_fund_futures = qdl.get('FRED/DFF', start_date = start_date, end_date= end_date, authtoken= api_key)

	fed_table = fed_fund_futures.join(effective_fund_futures, how= 'left', rsuffix= '')
	fed_table.columns = fed_reserve_list 
	fed_table.fillna(method = 'ffill', inplace = True )
	# Calculate the probability of a rate hike as the difference between rolling federal funds futures and the effective funds rate divided by the amount of the hike.
	# (Multiplied by 100 for percentage)
	fed_table['rate_hike_prob_25_basis'] = (((100 - fed_table['Federal_Funds_Futures']) - fed_table['Effective_Funds_Rate'])/ 0.25) * 100
	fed_table = fed_table.drop('Federal_Funds_Futures', axis= 1)
	fed_table = fed_table.drop('Effective_Funds_Rate', axis= 1)

	return fed_table 

def get_benchmark(benchmark_list, benchmark_quandl_list, num_days, end_date, api_key, shift):
	# Get returns of a benchmark asset 
	start_date = end_date - timedelta(num_days)
	benchmark_table = qdl.get(benchmark_quandl_list, start_date = start_date, end_date = end_date, authtoken= api_key)
	benchmark_table.columns = ['Benchmark']
	benchmark_returns = benchmark_table.pct_change(periods= shift).dropna() * 100 
	benchmark_returns.drop(benchmark_returns.index[:1], inplace=True)

	return benchmark_returns

def merge_with_technicals(currency_list, returns_table, fundamentals_table, RSI, MACD, Stochastics, beg_date, stoch_date):
	# Create empty list, will hold dataframes for all currencies
	dataframe_list = []
	for currency in currency_list:
		buildup_dataframe = DataFrame(returns_table[currency])
		buildup_dataframe = buildup_dataframe.join(fundamentals_table, how= 'left', rsuffix= '')
		buildup_dataframe = buildup_dataframe.join(RSI[currency], how= 'left', rsuffix= '_RSI')
		buildup_dataframe = buildup_dataframe.join(MACD[currency], how='left', rsuffix='_MACD')
		if beg_date > stoch_date:
			buildup_dataframe = buildup_dataframe.join(Stochastics[currency], how='left', rsuffix='_Stoch')
		dataframe_list.append(buildup_dataframe)

	return dataframe_list
	
if __name__ == "__main__":
	main()