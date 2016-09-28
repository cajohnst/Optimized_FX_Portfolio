''' Pull all data to be used in Optimize_FX_Portfolio, RSI_sample, MACD_sample, and futures_vs_spot (eventually pull fred economic reports) '''
import pandas as pd 
import quandl as qdl 
import numpy as np 
from pandas import Series, DataFrame
import datetime
from datetime import timedelta, date
import rollover_google_sheet
import RSI_sample
import fxstreet_scraper
import sklearn as sklearn
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from Optimize_FX_Portfolio import interest_rate

def main():
	auth_tok = "kz_8e2T7QchJBQ8z_VSi"

	num_days = 720

	# country_list = get_country_list()
	currency_list = get_currency_list()
	currency_quandl_list = get_currency_quandl_list()
	fed_list = get_fed_list()
	# fed_quandl_list = get_fed_quandl_list

	end_date = datetime.date.today()
	beg_date = end_date - timedelta(num_days)
	stoch_date = datetime.date(2016, 7, 15)


#############################################################################################################################
#Econ_Events data
	econ_calendar_today = pd.read_csv("/Users/cajohnst/Coding/event_calendar_today.csv", index_col = 'DateTime', parse_dates= True, infer_datetime_format = True)
	econ_calendar_full = pd.read_csv("/Users/cajohnst/Coding/Event_Calendar.csv", index_col = 'DateTime', parse_dates= True, infer_datetime_format= True, skip_blank_lines= True)  
	# econ_calendar_full = econ_calendar_full.ix[beg_date:end_date]
	econ_calendar_full.Consensus.fillna(econ_calendar_full.Previous, inplace = True)
	econ_calendar_full['Deviation'] = econ_calendar_full['Actual'] - econ_calendar_full['Consensus']

#############################################################################################################################
#RSI_sample data
	n = 14
	d = 3
	nslow = 26
	nfast = 12
	nema = 9
	q = 14
	ma_slow = 100
	ma_fast = 20

	list_high = [high.replace('1', '2') for high in currency_quandl_list]
	list_low = [low.replace('1', '3') for low in currency_quandl_list]
	max_lag = max(q, nslow, nfast, nema, ma_slow, ma_fast, n, d)

	#Pull data from quandl
	currency_table = get_currency_data(currency_list, currency_quandl_list, num_days, end_date, auth_tok)
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

	returns_table_pred = returns_table.copy()
	returns_table_pred = returns_table_pred.shift(periods = -1, axis = 0) * 100 

	# #Number of days for rollover data
	# no_days = 60
	# rollover_table = rollover_google_sheet.pull_data(no_days)
	# rollover_table = rollover_table / 10000
	# merge_table = merge_tables(returns_table, rollover_table)
	# merge_table = merge_table.dropna()
	fed_table = get_fed_data(fed_list, num_days, end_date, auth_tok)

	# Specialize data for events!
	economic_data_dict = get_economic_data_dict()
	all_fundamentals_table = query_past_economic_data(econ_calendar_today, econ_calendar_full, fed_table, economic_data_dict)
	# all_fundamentals_table = fed_table.join(econ_data, how= 'left', rsuffix= '')
	# print all_fundamentals_table

	regression_table = merge_with_technicals(currency_list, returns_table, all_fundamentals_table, RSI, macd, fast_stochastic, beg_date, stoch_date)
	reg_table, return_predictions = predict_returns(regression_table)

	return return_predictions

def query_past_economic_data(calendar_today, calendar_full, fed_table, economic_data_dict):
	for index, values in calendar_today.iterrows():
		country = values['Country']
		if country in economic_data_dict:
			event_name = values['Name']
			if event_name in economic_data_dict[country]:
				pull_events = calendar_full[(calendar_full['Country'] == country) & (calendar_full['Name'] == event_name)]
				fed_table = fed_table.join(pull_events['Deviation'], how = 'left', rsuffix = ' of {0}'.format(event_name))
				fed_table = fed_table.fillna(value = 0)

	return fed_table 

# def get_country_list():
# 	country_list = ['Australia', 'Canada', 'China', 'Japan', 'Mexico', 'New Zealand', 'South Africa', 'Singapore', 'United Kingdom', 'United States']
# 	return country_list 

def get_currency_quandl_list():
	currency_quandl_list = ['CURRFX/MXNUSD.1', 'CURRFX/USDCAD.1', 'CURRFX/NZDUSD.1', 'CURRFX/USDHKD.1', 'CURRFX/USDJPY.1', 'CURRFX/USDSGD.1', 'CURRFX/GBPUSD.1', 'CURRFX/USDZAR.1',
							'CURRFX/AUDUSD.1', 'CURRFX/EURUSD.1']
	return currency_quandl_list

def get_currency_list():
	currency_list = ['MXN/USD', 'USD/CAD', 'NZD/USD', 'USD/HKD', 'USD/JPY', 'USD/SGD', 'GBP/USD', 'USD/ZAR', 'AUD/USD', 'EUR/USD']
	return currency_list 

# def get_us_quandl_list():
# 	us_quandl_list = ['fred/cpiaucsl', 'fred/cpilfesl', 'fred/payems', 'umich/soc1', 'bkrhughes/rigs_by_state_totalus_total', 'fred/napm', 'fred/dgorder', 'fred/rsafs', 'fred/icsa', 'adp/empl_sec', 'fred/gdp', 'fred/unrate', 'fred/m2', 'fred/houst', 'fred/dgs10']
# 	return us_quandl_list


def get_economic_data_dict():
	# Dictonary keys are the country name
	# Key values are tuples structured as ([list of eventnames from fxstreet], [list of eventnames from quandl])
	economic_data_dict = {
	'United States': 
		['Consumer Price Index (YoY)', 'Consumer Price Index Ex Food & Energy (YoY)', 'Nonfarm Payrolls', 'Reuters/Michigan Consumer Sentiment Index', 'Baker Hughes US Oil Rig Count', 'Durable Goods Orders', 'Durable Goods Orders ex Transportation', 'Retail Sales (MoM)', 'Initial Jobless Claims', 'ADP', 'Gross Domestic Product Annualized', 'Unemployment Rate', 'M2', 'Housing Starts (MoM)', 'Building Permits (MoM)', '10-Year Note Auction', 
		'OPEC Meeting', 'S&P/Case-Shiller Home Price Indices (YoY)', 'Markit Services PMI', 'Markit PMI Composite', 'Consumer Confidence', 'Dallas Fed Manufacturing Business Index', ]
	,
	# 'Japan':
	# 	([],[]),
	}

	# return_dictionary = dict()
	# for country, (fxstreet_names, quandl_names) in economic_data_dict.iteritems():
	# 	return_dictionary[country] = {}
	# 	for index, name in enumerate(fxstreet_names):
	# 		return_dictionary[country][name] = quandl_names[index]

	return economic_data_dict 

def get_fed_list():
	fed_list = ['Federal_Funds_Futures', 'Effective_Funds_Rate']
	return fed_list 


# def pull_economic_data(returns_table, event_calendar, us_dict, fed_dict, num_days):
# 	today = datetime.date.today()
# 	end_date = today
# 	start_date = end_date - timedelta(num_days)

# 	econ_table = returns_table.copy()
# 	fed_table = None

# 	if 'United States' in event_calendar['country']:

# 		for key, quandl_name in us_dict.items():
# 			if key in event_list:
# 				current_column = qdl.get(quandl_name, start_date= start_date, end_date= end_date, authtoken= api_key)
# 				if 'Consumer Price Index (YoY)' in current_column.columns:
# 					current_column = current_column.pct_change(shift= 12).dropna()
# 				if 'Consumer Price Index Ex Food & Energy (YoY)' in current_column.columns:
# 					current_column = current_column.pct_change(shift = 12).dropna()
# 				if 'Consumer Price Index (MoM)' in current_column.columns:
# 					current_column = current_column.pct_change().dropna()
# 				if 'Consumer Price Index Ex Food & Energy (MoM)' in current_column.columns:
# 					current_column = current_column.pct_change().dropna()
# 				else:
# 					current_column = current_column.diff(periods = 1)
# 				current_column.columns = ['key']
# 				if econ_table = None:
# 					econ_table= current_column
# 				else:
# 					econ_table = econ_table.join(current_column, how= 'left', rsuffix = '')

# 	return econ_table

def get_currency_data(currency_list, currency_quandl_list, num_days, end_date, api_key):
	# Calculate dates
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

	return data_table 

def get_fed_data(fed_reserve_list, num_days, end_date, api_key):
	# Calculate dates
	start_date = end_date - timedelta(num_days)

	fed_fund_futures = qdl.get('CHRIS/CME_FF1.6', start_date= start_date, end_date= end_date, authtoken= api_key)
	effective_fund_futures = qdl.get('FRED/DFF', start_date = start_date, end_date= end_date, authtoken= api_key)

	fed_table = fed_fund_futures.join(effective_fund_futures, how= 'left', rsuffix= '')
	fed_table.columns = fed_reserve_list 
	fed_table.fillna(method = 'ffill', inplace = True )
	fed_table['rate_hike_prob_25_basis'] = (((100 - fed_table['Federal_Funds_Futures']) - fed_table['Effective_Funds_Rate'])/ 0.25) * 100
 	fed_table = fed_table.drop('Federal_Funds_Futures', axis= 1)
	fed_table = fed_table.drop('Effective_Funds_Rate', axis= 1)

	return fed_table 

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

def predict_returns(regression_table):
	pred_pbar = []
	r_squares = []
	for dataframe in regression_table:
		dataframe = dataframe.dropna()
		clf = Ridge(alpha=0.1, normalize = True)
		ret = dataframe.iloc[:-1, 0:1].copy()
		X = dataframe.iloc[:-1, 1:].copy()
		clf.fit(X, ret)
		x_test = dataframe.iloc[-1:, 1:].copy()
		y_test = dataframe.iloc[-1:, 0:1].copy()
		pred = clf.predict(x_test)
		pred_pbar_to_float = float(pred[0])
		pred_pbar.append(pred_pbar_to_float)
		r_square_to_float = float(clf.score(x_test, y_test))
		r_squares.append(r_square_to_float)
	new_list = []
	for pred in pred_pbar:
		short = -pred 
		new_list.append(pred)
		new_list.append(short)
	new_list.append(interest_rate)
	print new_list 

	# print 'pbar predictions'
	# print pred_pbar
	# print 'rsquared'
	# print r_squares
	return regression_table, pred_pbar, new_list


	
if __name__ == "__main__":
	main()