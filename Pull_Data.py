''' Pull all data to be used in Optimize_FX_Portfolio, RSI_sample, MACD_sample, and futures_vs_spot (eventually pull fred economic reports) '''
import settings as sv 
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
import argparse

def main():
	currency_list = sv.get_currency_list()
	currency_quandl_list = sv.get_currency_quandl_list()
	fed_list = get_fed_list() 
	# Calculate beginning date
	beg_date = sv.end_date  - timedelta(sv.num_days_regression)

#############################################################################################################################
	# Import events data into pandas dataframe
	econ_calendar_full = pd.DataFrame(fxstreet_google_sheet.pull_data(sv.num_days_regression))
	# If no consensus value exists, use the previous value to fill its place.
	econ_calendar_full.Consensus.fillna(econ_calendar_full.Previous, inplace = True)
	# Create column 'Deviation' as the actual value on the data release subtracting the market expectation
	econ_calendar_full['Deviation'] = econ_calendar_full['Actual'] - econ_calendar_full['Consensus']

	#Take today's Economic Events from FXstreet_scraper and format
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
	# For stochastic data, import daily highs and lows into dataframe from quandl
	# Highs and Lows are unreliable before sv.stoch_date (and need to be continually monitored)
	list_high = [high.replace('1', '2') for high in currency_quandl_list]
	list_low = [low.replace('1', '3') for low in currency_quandl_list]
	# Maximum of the 'rolling' periods
	max_lag = max(sv.q, sv.nslow, sv.nfast, sv.nema, sv.ma_slow, sv.ma_fast, sv.n, sv.d)

	#Pull data from quandl
	currency_table = get_currency_data(currency_list, currency_quandl_list, sv.num_days_regression, sv.end_date , sv.auth_tok)

	live_rates = import_spot_test.main()
	currency_table = currency_table.append(live_rates)

	#Get daily lows from quandl for stochastic oscillator
	low_table = get_currency_data(currency_list, list_low, sv.num_days_regression, sv.end_date , sv.auth_tok)
	#Get daily highs from quandl for stochastic oscillator
	high_table = get_currency_data(currency_list, list_high, sv.num_days_regression, sv.end_date , sv.auth_tok)

	# #Calculate RSI for all currency pairs in currency_table
	RSI = RSI_sample.RSI_Calc(currency_table, sv.q)

	#Calculate exponentially weighted moving averages and MACD
	emaslow, emafast, macd = RSI_sample.get_MACD(currency_table, nslow= sv.nslow, nfast = sv.nfast)
	ema9 = RSI_sample.moving_average(macd, sv.nema, type = 'exponential')

	#Calculate stochastics
	fast_stochastic, slow_stochastic = RSI_sample.get_stochastic(currency_table, low_table, high_table, sv.n, sv.d)

	# #Calculate simple moving averages
	ma_f = RSI_sample.moving_average(currency_table, sv.ma_fast, type='simple')
	ma_s = RSI_sample.moving_average(currency_table, sv.ma_slow, type='simple')

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
	returns_table = currency_table.pct_change(periods= sv.shift).dropna()
	returns_table.drop(returns_table.index[:1], inplace=True)
	returns_table = 100 * sv.leverage * returns_table 

	fed_table = get_fed_data(fed_list, sv.num_days_regression, sv.end_date , sv.auth_tok)

	# Specialize data for events! Pull all historical data from event calendar which matches name in econ data dictionary.
	economic_data_dict = get_economic_data_dict()
	all_fundamentals_table = query_past_economic_data(econ_calendar_today, econ_calendar_full, fed_table, economic_data_dict)
	#Merge the calendar data with the columns of technicals
	regression_table = merge_with_technicals(currency_list, returns_table, all_fundamentals_table, RSI, macd, fast_stochastic, beg_date, sv.stoch_date)
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

def get_economic_data_dict():
	# Dictonary keys are the country name
	# Key values are tuples structured as ([list of eventnames from fxstreet], [list of eventnames from quandl])
	economic_data_dict = {
	'United States': 
		['Consumer Price Index (YoY)', 'Consumer Price Index Ex Food & Energy (YoY)', 'Nonfarm Payrolls', 'Reuters/Michigan Consumer Sentiment Index', 'Baker Hughes US Oil Rig Count', 'Durable Goods Orders', 'Durable Goods Orders ex Transportation', 'Retail Sales (MoM)', 'Initial Jobless Claims', 'ADP Employment Change', 'Gross Domestic Product Annualized', 'Unemployment Rate', 'M2', 'Housing Starts (MoM)', 'Building Permits (MoM)', '10-Year Note Auction', 
		'EIA Crude Oil Stocks change', 'S&P/Case-Shiller Home Price Indices (YoY)', 'Markit Services PMI', 'Markit PMI Composite', 'Consumer Confidence', 'Dallas Fed Manufacturing Business Index', 'ISM Prices Paid', 'ISM Manufacturing PMI', 'Markit Manufacturing PMI', 'Construction Spending (MoM)', 'Trade Balance', 'ISM Non-Manufacturing PMI', 'Factory Orders (MoM)', 'Average Hourly Earnings (MoM)', 'Average Hourly Earnings (YoY)', 'Labor Force Participation Rate',
		'Consumer Credit Change', 'Labor Market Conditions Index', 'Export Price Index (MoM)', 'Import Price Index (MoM)', 'Export Price Index (YoY)', 'Import Price Index (YoY)', 'Retail Sales ex Autos (MoM)', 'Industrial Production (MoM)', 'Consumer Price Index (MoM)', 'Consumer Price Index Ex Food & Energy (MoM)', 'NAHB Housing Market Index', 'Consumer Price Index n.s.a (MoM)', 'Consumer Price Index Core s.a', 'Philadelphia Fed Manufacturing Survey',
		'CB Leading Indicator (MoM)', 'Chicago Fed National Activity Index ']
	,
	'Japan':
		['National Consumer Price Index (YoY)', 'Foreign investment in Japan stocks', 'Foreign bond investment', 'Unemployment Rate', 'Industrial Production (MoM)', 'Industrial Production (YoY)', 'Tankan Non - Manufacturing Index', 'Tankan Large All Industry Capex', 'Tankan Large Manufacturing Outlook', 'Tankan Large Manufacturing Index', 'Tankan Non - Manufacturing Outlook', 'Nikkei Manufacturing PMI', 'Vehicle Sales (YoY)', 'Tertiary Industry Index (MoM)',
		'All Industry Activity Index (MoM)', 'Exports (YoY)', 'Imports (YoY)', 'Merchandise Trade Balance Total', 'Adjusted Merchandise Trade Balance']
	,
	'European Monetary Union':
		['Unemployment Rate', 'Consumer Price Index (YoY)', 'Consumer Price Index - Core (YoY)', 'Markit Manufacturing PMI', 'Producer Price Index (MoM)', 'Producer Price Index (YoY)', 'Markit Services PMI', 'ZEW Survey - Economic Sentiment', 'Industrial Production w.d.a. (YoY)', 'Industrial Production s.a. (MoM)', 'Trade Balance n.s.a.', 'Trade Balance s.a.', 'Consumer Price Index (MoM)', 'Consumer Price Index - Core (MoM)', 'ECB Interest Rate Decision', 'ECB deposit rate decision', 'Markit PMI Composite']
	,
	'Germany':
		['Markit Manufacturing PMI', '10-y Bond Auction', 'Exports (MoM)', 'Trade Balance s.a.', 'Imports (MoM)', 'Current Account n.s.a.', 'ZEW Survey - Current Situation', 'ZEW Survey - Economic Sentiment', 'Wholesale Price Index (YoY)', 'Wholesale Price Index (MoM)', 'Harmonised Index of Consumer Prices (MoM)', 'Consumer Price Index (YoY)', 'Harmonised Index of Consumer Prices (YoY)', 'Consumer Price Index (MoM)', 'Producer Price Index (YoY)', 'Producer Price Index (MoM)', 'Markit PMI Composite', 'Markit Services PMI']
	,
	'Australia':
		['TD Securities Inflation (YoY)', 'TD Securities Inflation (MoM)', 'RBA Interest Rate Decision', 'Retail Sales s.a. (MoM)', 'AiG Performance of Mfg Index', 'Imports', 'Exports', 'Trade Balance', 'AiG Performance of Construction Index', 'Home Loans', 'Investment Lending for Homes', 'Westpac Consumer Confidence', 'Consumer Inflation Expectation', 'Participation Rate', 'Unemployment Rate s.a.', 'Employment Change s.a.']
	,
	'Canada':
		['RBC Manufacturing PMI', 'Unemployment Rate', 'Participation rate', 'Net Change in Employment', 'Ivey Purchasing Managers Index', 'Ivey Purchasing Managers Index s.a', 'Housing Starts s.a (YoY)', 'BoC Interest Rate Decision']
	,
	'New Zealand':
		['Electronic Card Retail Sales  (MoM)', 'Electronic Card Retail Sales (YoY)', 'Business NZ PMI', 'Consumer Price Index (QoQ)', 'Consumer Price Index (YoY)']
	,
	'China':
		['FDI - Foreign Direct Investment (YTD) (YoY)', 'Trade Balance CNY', 'Imports (YoY)', 'Exports (YoY)', 'Trade Balance USD', 'Producer Price Index (YoY)', 'Consumer Price Index (MoM)', 'Consumer Price Index (YoY)', 'Retail Sales (YoY)', 'Industrial Production (YoY)', 'Gross Domestic Product (YoY)', 'Gross Domestic Product (QoQ)']
	,
	'United Kingdom':
		['Markit Manufacturing PMI', 'PMI Construction', 'Manufacturing Production (YoY)', 'Manufacturing Production (MoM)', 'Industrial Production (MoM)', 'Industrial Production (YoY)', 'RICS Housing Price Balance', 'Producer Price Index - Output (MoM) n.s.a', 'Producer Price Index - Output (YoY) n.s.a', 'PPI Core Output (MoM) n.s.a', 'Producer Price Index - Input (YoY) n.s.a', 'Producer Price Index - Input (MoM) n.s.a', 'PPI Core Output (YoY) n.s.a ', 
		'Consumer Price Index (YoY)', 'Consumer Price Index (MoM)', 'Core Consumer Price Index (YoY)', 'Consumer Price Index (MoM)', 'Core Consumer Price Index (YoY)', 'Retail Sales (MoM)', 'Retail Sales (YoY)', 'Retail Sales ex-Fuel (MoM)', 'Retail Sales ex-Fuel (YoY)', 'Gross Domestic Product (QoQ)', 'Gross Domestic Product (YoY)']
	, 
	'Italy':
		['Markit Manufacturing PMI']
	, 
	'Switzerland':
		['Real Retail Sales (YoY)', 'Consumer Price Index (MoM)', 'Consumer Price Index (YoY)', 'Unemployment Rate s.a (MoM)', 'ZEW Survey - Expectations']
	,
	'France':
		['Markit Manufacturing PMI']
	,
	'Spain':
		['Markit Manufacturing PMI']
	}

	return economic_data_dict 

def get_fed_list():
	fed_list = ['Federal_Funds_Futures', 'Effective_Funds_Rate']
	return fed_list 


def get_currency_data(currency_list, currency_quandl_list, num_days_regression, end_date , api_key):
	# Calculate dates to begin and end
	start_date = end_date  - timedelta(num_days_regression)

	# Initialize data table
	data_table = None
	# Run through currencies, first assignment is initialized
	# Anything past first currency is joined into table 
	for currency in currency_quandl_list:
		current_column = qdl.get(currency, start_date= start_date, end_date = end_date , authtoken= api_key)
		current_column.columns = [currency]
		if data_table is None:
			data_table = current_column
		else:
			data_table = data_table.join(current_column, how= 'left', rsuffix= '')
	data_table.columns = currency_list 
	if 'USD/MXN' in currency_list:
		data_table['USD/MXN'] = 1 / data_table['USD/MXN']
	return data_table 

def get_fed_data(fed_reserve_list, num_days_regression, end_date , api_key):
	# Calculate dates
	start_date = end_date  - timedelta(num_days_regression)
	# Get Federal Funds Futures data and Effective Funds Futures
	fed_fund_futures = qdl.get('CHRIS/CME_FF1.6', start_date= start_date, end_date = end_date , authtoken= api_key)
	effective_fund_futures = qdl.get('FRED/DFF', start_date = start_date, end_date = end_date , authtoken= api_key)

	fed_table = fed_fund_futures.join(effective_fund_futures, how= 'left', rsuffix= '')
	fed_table.columns = fed_reserve_list 
	fed_table.fillna(method = 'ffill', inplace = True )
	# Calculate the probability of a rate hike as the difference between rolling federal funds futures and the effective funds rate divided by the amount of the hike.
	# (Multiplied by 100 for percentage)
	fed_table['rate_hike_prob_25_basis'] = (((100 - fed_table['Federal_Funds_Futures']) - fed_table['Effective_Funds_Rate'])/ 0.25) * 100
	fed_table = fed_table.drop('Federal_Funds_Futures', axis= 1)
	fed_table = fed_table.drop('Effective_Funds_Rate', axis= 1)

	return fed_table 

def get_benchmark(benchmark_list, benchmark_quandl_list, num_days_regression, end_date , api_key, shift):
	# Get returns of a benchmark asset 
	start_date = end_date  - timedelta(num_days_regression)
	benchmark_table = qdl.get(benchmark_quandl_list, start_date = start_date, end_date  = end_date , authtoken= api_key)
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