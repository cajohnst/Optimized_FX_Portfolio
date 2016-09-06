import quandl as qdl
import pandas as pd
import numpy as np
import pandas.io.data as web
from pandas.io.data import DataReader
from pandas import Series, DataFrame, ExcelWriter
import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

def main():
	currency_list = get_currency_list()
	futures_list = get_futures_list()
	num_days = 365
	#Compute returns with shift delay
	shift = 1
	#Compute returns
	# spot_table = get_daily_spot(currency_list, num_days, shift)

	# futures_table = get_daily_futures(futures_list, num_days)

	spot_table = DataFrame.from_csv('spots_table.txt')
	futures_table = DataFrame.from_csv('futures_table.txt')

	spot_table = spot_table.pct_change()
	futures_table = futures_table.pct_change()
	futures_table.columns = ['EUR/USD', 'GBP/USD', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'USD/JPY']
	difference_table = futures_table - spot_table
	difference_table = difference_table.dropna().add_suffix('_dif')
	merge_table = spot_table.join(difference_table, how = 'inner')


	for col in range(6):
		for index, date in enumerate(merge_table.index):
			dif_val = merge_table.iloc[index][futures_table.columns[col]]
			plt.plot(date, dif_val, 'bo')
			plt.xlabel('Date')
			plt.ylabel('Returns')
			plt.title('{0} Spot vs. Futures Returns Divergence'.format(spot_table.columns[col]))
		plt.show()

	for col in spot_table.columns:
		ols_result = sm.OLS(merge_table[col], merge_table[col + '_dif']).fit()
		print('Parameters:',ols_result.params)
		print('R^2:', ols_result.rsquared)
		print(ols_result.summary())

		prstd, iv_l, iv_u = wls_prediction_std(ols_result)

		fig, ax = plt.subplots(figsize=(8,6))

		ax.plot(merge_table[col + '_dif'], merge_table[col], 'o', label="data")
		ax.plot(merge_table[col + '_dif'], ols_result.fittedvalues, 'y-', label="OLS")
		ax.plot(merge_table[col + '_dif'], iv_u, 'r--', label= "Upper")
		ax.plot(merge_table[col + '_dif'], iv_l, 'r--', label= "Lower")
		ax.set_xlabel('Percent Divergence')
		ax.set_ylabel('Percent Change in Next Day')
		ax.legend(loc='best');
		plt.title(col + ' Divergence between Futures and Spot vs. Next Day Spot')
		plt.show()

def get_currency_list():
	currency_list = ['DEXUSEU', 'DEXUSUK', 'DEXUSAL', 'DEXCAUS', 'DEXUSNZ', 'DEXJPUS']
	return currency_list

def get_futures_list():
	futures_list = ['CHRIS/CME_EC1.1', 'CHRIS/CME_BP1.1', 'CHRIS/CME_AD1.1', 'CHRIS/CME_CD3.1', 'CHRIS/CME_NE1.1', 'CHRIS/CME_JY1.1']
	return futures_list

def get_daily_spot(currency_list, num_days, shift):
	# Calculate dates
	today = datetime.date.today()
	start_date = today - timedelta(num_days)

	# Initialize data table
	data_table = None
	# Run through currencies, first assignment is initialized
	# Anything past first currency is joined into table
	for currency in currency_list:
		current_column = DataReader(currency, 'fred', start_date, today)
		if data_table is None:
			data_table = current_column
		else:
			data_table = data_table.join(current_column)

	spot_table = data_table.dropna()
	spot_table.columns = ['EUR/USD', 'GBP/USD', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'USD/JPY']
	return spot_table

def get_daily_futures(futures_list, num_days):
	today = datetime.date.today()
	start_date = today - timedelta(num_days)
	end_date = today
	#Initialize data table
	futures_table = None

	for currency in futures_list:
		current_column = qdl.get(currency, start_date= start_date, end_date= end_date)
		if futures_table is None:
			futures_table = current_column
		else:
			futures_table = futures_table.join(current_column, how= 'left', rsuffix= 'F_')

	futures_table.columns = ['F_EUR/USD', 'F_GBP/USD', 'F_AUD/USD', 'F_USD/CAD', 'F_NZD/USD', 'F_USD/JPY']

	if 'F_USD/CAD' in futures_table.columns:
		futures_table['F_USD/CAD'] = 1/ futures_table['F_USD/CAD']

	if 'F_USD/JPY' in futures_table.columns:
		futures_table['F_USD/JPY'] = 1 / (futures_table['F_USD/JPY'] / 1000000)

	return futures_table

if __name__ == "__main__":
	main()