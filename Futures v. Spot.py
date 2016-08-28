import quandl as qdl
import pandas as pd
import numpy as np
import pandas.io.data as web
from pandas.io.data import DataReader
from pandas import Series, DataFrame, ExcelWriter
import datetime
from datetime import date, timedelta

def main():
	currency_list = get_currency_list()
	num_days = 365
	#Compute returns with shift delay
	shift = 1
	#Compute returns
	returns_table = get_daily_spot(currency_list, num_days, shift)

	futures_table = get_daily_futures(currency_list, num_days)
	print futures_table

def get_currency_list():
	currency_list = ['DEXUSEU']
	return currency_list

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

	# Convert price data to returns and delete NaNs
	spot_table = data_table.pct_change(periods= shift).dropna()
	# Specify number of days to shift

	spot_table.drop(returns_table.index[:1], inplace=True)
	return spot_table

def get_daily_futures(currency_list, num_days):
	today = datetime.date.today()
	start_date = today - timedelta(num_days)
	end_date = today -timedelta(7)
	#Initialize data table
	futures_data = None
	futures_data = quandl.get("CHRIS/CME_EC1", start_date= start_date, end_date= end_date)
	return futures_data

if __name__ == "__main__":
	main()