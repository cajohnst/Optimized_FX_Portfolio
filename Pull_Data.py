''' Pull all data to be used in Optimize_FX_Portfolio, RSI_sample, MACD_sample, and futures_vs_spot (eventually pull fred economic reports) '''
import pandas as pd 
import quandl as qdl 
import numpy as np 
from pandas import Series, DataFrame
import datetime
from datetime import date, timedelta
import rollover_google_sheet
import RSI_sample

def main():
	auth_tok = "kz_8e2T7QchJBQ8z_VSi"
	currency_dict = currency_dict()
	us_dictionary = us_economic_dict()
	fed_dicionary = fed_dict()

	n = 14
	d = 3
	nslow = 26
	nfast = 12
	nema = 9
	q = 14

	list_high = [high.replace('1', '2') for high in currency_dict]
	list_low = [low.replace('1', '3') for low in currency_dict]
	max_lag = max(q, nslow, nfast, nema, ma_slow, ma_fast, n, d)

	if 'GDP' in pull_list:
		num_days = 1825 + max_lag
	else:
		num_days = 750 + max_lag

	#Pull data from quandl
	currency_table = get_currency_data(currency_dict, num_days, auth_tok)
	#Get daily lows from quandl for stochastic oscillator
	low_table = get_currency_data(list_low, num_days, auth_tok)
	#Get daily highs from quandl for stochastic oscillator
	high_table = get_currency_data(list_high, num_days, auth_tok)

	# #Calculate RSI for all currency pairs in currency_table
	RSI = RSI_sample.RSI_Calc(currency_table, q)

	#Calculate exponentially weighted moving averages and MACD
	emaslow, emafast, macd = RSI_sample.get_MACD(currency_table, nslow= nslow, nfast = nfast)

	#Calculate stochastics
	fast_stochastic, slow_stochastic = RSI_sample.get_stochastic(currency_table, low_table, high_table, n, d)

	RSI = drop_rows(RSI, max_lag)
	emaslow= drop_rows(emaslow, max_lag)
	emafast= drop_rows(emafast, max_lag)
	macd = drop_rows(macd, max_lag)
	fast_stochastic = drop_rows(fast_stochastic, max_lag)
	currency_table = drop_rows(currency_table, max_lag)

	
	econ_calendar, pull_list = pull_econ_calendar()
	econ_data = pull_economic_data(returns_table, pull_list, us_dictionary, fed_dictionary, num_days)

	if 'Federal Reserve' | 'FOMC' in pull_list:
		fed_dates = get_fed_dates()
		fed_dates_table = econ_data[econ_data.index.isin(fed_dates)]


	returns_table = get_currency_data(currency_dict, num_days, shift, auth_tok)

	#Number of days for rollover data
	no_days = 60
	rollover_table = rollover_google_sheet.pull_data(no_days)
	rollover_table = rollover_table / 10000
	merge_table = merge_tables(returns_table, rollover_table)
	merge_table = merge_table.dropna()

	regression_table = merge_with_technicals(econ_data, RSI, macd, fast_stochastic)

	return calendar, regression_table

def pull_econ_calendar():
	''' import calendar from fxstreet.com '''

	calendar = calendar[(calendar['date'] == 'today') & (calendar['volatility'] >= 2)]
	pull_list = event_frame['event_column'].to_list()
	return calendar, pull_list


def currency_dict():
	currency_dict = {'MXN/USD':'CURRFX/MXNUSD.1', 'USD/CAD':'CURRFX/USDCAD.1', 'NZD/USD':'CURRFX/NZDUSD.1', 'USD/HKD':'CURRFX/USDHKD.1', 
						'USD/JPY':'CURRFX/USDJPY.1', 'USD/SGD':'CURRFX/USDSGD.1', 'GBP/USD':'CURRFX/GBPUSD.1', 'USD/ZAR':'CURRFX/USDZAR.1',
						 'AUD/USD':'CURRFX/AUDUSD.1', 'EUR/USD':'CURRFX/EURUSD.1'}
	return currency_dict

def us_economic_dict():
	us_econ_dict = {'Consumer Price Index (YoY)':'fred/cpiaucsl', 'Consumer Price Index Ex Food & Energy (YoY)': 'fred/cpilfesl', 'Nonfarm Payrolls': 'fred/payems', 'Reuters/Michigan Consumer Sentiment Index': 'umich/soc1', 
					'Baker Hughes US Oil Rig Count': 'bkrhughes/rigs_by_state_totalus_total', 'PMI Composite': 'fred/napm', 'Durable Goods': 'fred/dgorder', 
					'Retail Sales (MoM)': 'fred/rsafs', 'Initial Jobless Claims': 'fred/icsa', 'ADP': 'adp/empl_sec', 'GDP': 'fred/gdp', 
					'Unemployment Rate': 'fred/unrate', 'M2': 'fred/m2', 'Housing Starts': 'fred/houst', '10-Year Yield': 'fred/dgs10'}

	return us_econ_dict

def euro_dict():
	euro_econ_dict = {}
	return euro_econ_dict

def jpy_dict():
	jpy_econ_dict = {}
	return jpy_econ_dict

def gbp_dict():
	gbp_econ_dict = {}
	return gbp_econ_dict

def aud_dict():
	aud_econ_dict = {}
	return aud_econ_dict

def cad_dict():
	cad_econ_dict = {}
	return cad_econ_dict 

def cny_dict():
	cny_econ_dict = {}
	return cny_econ_dict

def nzd_dict():
	nzd_econ_dict = {}
	return nzd_econ_dict

def fed_dict():
	fed_dict = {'Federal Funds Futures': 'CHRIS/CME_FF1', 'Effective Funds Rate': 'FRED/DFF'}
	return fed_dict


def pull_economic_data(returns_table, event_calendar, us_dict, fed_dict, num_days):
	today = datetime.date.today()
	end_date = today
	start_date = end_date - timedelta(num_days)

	econ_table = returns_table.copy()
	fed_table = None

	if 'United States' in event_calendar['country']:

		for key, quandl_name in us_dict.items():
			if key in event_list:
				current_column = qdl.get(quandl_name, start_date= start_date, end_date= end_date, authtoken= api_key)
				if 'Consumer Price Index (YoY)' in current_column.columns:
					current_column = current_column.pct_change(shift= 12).dropna()
				if 'Consumer Price Index Ex Food & Energy (YoY)' in current_column.columns:
					current_column = current_column.pct_change(shift = 12).dropna()
				if 'Consumer Price Index (MoM)' in current_column.columns:
					current_column = current_column.pct_change().dropna()
				if 'Consumer Price Index Ex Food & Energy (MoM)' in current_column.columns:
					current_column = current_column.pct_change().dropna()
				else:
					current_column = current_column.diff(periods = 1)
				current_column.columns = ['key']
				if econ_table = None:
					econ_table= current_column
				else:
					econ_table = econ_table.join(current_column, how= 'left', rsuffix = '')

		econ_table = econ_table.fillna(method = 'ffill')
	if 'Federal Reserve' | 'FOMC' not in event_list:
		for key2, quandl_name_fed in fed_dict.items():
			current_column = qdl.get(quandl_name_fed, start_date= start_date, end_date = end_date, authtoken = api_key)
			current_column.columns = ['key2']
			if fed_table = None:
				fed_table = current_column
			else:
					fed_table = fed_table.join(current_column, how= 'left', rsuffix = '')
		fed_table['rate_hike_prob_25_basis'] = (fed_table['Federal Funds Futures'] - fed_table['Effective Funds Rate'])/ 0.25
		fed_table['rate_hike_prob_50_basis'] = (fed_table['Federal Funds Futures'] - fed_table['Effective Funds Rate'])/ 0.5
		fed_table = fed_table.drop('Federal Funds Futures')
		fed_table = fed_table.drop('Effective Funds Rate')
		econ_table = econ_table.join(fed_table, how= 'left', rsuffix = '')

	return econ_table

def merge_with_technicals(econ_table, RSI, MACD, Stochastics):

	reg_table = econ_table.join(RSI, how= 'left', rsuffix= '')
	reg_table = reg_table.join(MACD, how= 'left', rsuffix= '')
	reg_table = reg_table.join(Stochastics, how= 'left', rsuffix= '')

	return reg_table 

def get_fed_dates():
	fed_speech_fomc_minutes_dates = ['01/20/2015', '01/30/2015', '02/04/2015', '02/09/2015', '02/18/2015', '02/27/2015', '03/03/2015', '03/23/2015', '03/27/2015', 
	'03/30/2015', '04/02/2015', '04/08/2015', '04/30/2015', '05/05/2015', '05/06/2015', '05/14/2015', '05/20/2015','05/21/2015', '05/22/2015', '05/26/2015', 
	'06/1/2015', '06/2/2015', '06/24/2015', '06/25/2015', '06/30/2015', '07/01/2015', '07/08/2015', '07/09/2015', '07/10/2015', '08/03/2015', '08/19/2015', 
	'08/29/2015', '09/24/2015', '09/28/2015', '09/30/2015', '10/02/2015','10/08/2015', '10/11/2015', '10/12/2015', '10/19/2015', '10/20/2015', '11/04/2015', 
	'11/05/2015', '11/06/2015', '11/12/2015', '11/17/2015', '11/18/2015', '11/19/2015', '11/20/2015', '12/01/2015', '12/02/2015', '12/03/2015', '01/03/2016',
	'01/06/2016', '02/01/2016', '02/10/2016', '02/17/2016', '02/23/2016', '02/26/2016', '03/07/2016', '03/29/2016', '04/06/2016', '04/14/2016', '05/18/2016', 
	'05/19/2016', '05/20/2016', '05/26/2016', '06/03/2016','06/06/2016', '06/21/2016', '06/22/2016', '06/28/2016', '06/28/2016', '07/06/2016', '07/12/2016', 
	'08/17/2016', '08/21/2016', '08/26/2016', '09/12/2016']

		return fed_speech_fomc_minutes_dates

def get_currency_data(currency_list, num_days, shift, api_key):
	# Calculate dates
	end_date = datetime.date.today()
	start_date = end_date - timedelta(num_days)

	# Initialize data table
	data_table = None
	# Run through currencies, first assignment is initialized
	# Anything past first currency is joined into table
	for currency, quandl_currency in currency_dict:
		current_column = qdl.get(quandl_currency, start_date= start_date, end_date= end_date, authtoken= api_key)
		current_column.columns = ['currency']
		if data_table is None:
			data_table = current_column
		else:
			data_table = data_table.join(current_column, how= 'left', rsuffix= ' ')


	# Convert price data to returns and delete NaNs
	returns_table = data_table.pct_change(periods= shift).dropna()
	# Specify number of days to shift

	returns_table.drop(returns_table.index[:1], inplace=True)

	return returns_table

def merge_tables(returns_table, rollover_table):
	merge_table = pd.DataFrame(columns=rollover_table.columns)
	for index, column in enumerate(returns_table):
		merge_table[merge_table.columns[(index * 2)]] = -1 * returns_table.ix[:,index]
		merge_table[merge_table.columns[(index * 2) + 1]] = returns_table.ix[:,index]
	merge_table = merge_table + rollover_table
	return merge_table

	
if __name__ == "__main__":
	main()