import pandas as pd 
from pandas.io.data import DataReader
import quandl as qdl
import datetime
from datetime import date, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.finance as finance
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def main():
	auth_tok = "kz_8e2T7QchJBQ8z_VSi"
	num_days = 252
	to_date = datetime.date.today()
	currency_list = get_currency_list()
	currency_table = get_currency_data(currency_list, num_days, auth_tok)

	#q = avg. periods for gain/loss
	q = 14
	Overbought = 70
	Oversold = 30

	RSI = RSI_Calc(currency_table, q)
	print RSI
	# chart_data = currency_table.join(RSI, how = 'inner', rsuffix = ' ')

	# for stock in stock_list:

	# 	plt.rc('axes', grid=True)
	# 	plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)

	# 	textsize = 9
	# 	left, width = 0.1, 0.8
	# 	rect1 = [left, 0.7, width, 0.2]
	# 	rect2 = [left, 0.3, width, 0.4]

	# 	fig = plt.figure(facecolor='white')
	# 	axescolor = '#f6f6f6'  # the axes background color

	# 	ax1 = fig.add_axes(rect1, axisbg=axescolor)  # left, bottom, width, height
	# 	ax2 = fig.add_axes(rect2, axisbg=axescolor, sharex=ax1)

	# 	ax1.plot(dates, RSI_vals, color='blue')
	# 	ax1.axhline(Overbought, color='red')
	# 	ax1.axhline(Oversold, color='green')
	# 	ax1.fill_between(dates, RSI_vals, 70, where=(RSI >= 70), facecolor='red', edgecolor='red')
	# 	ax1.fill_between(dates, RSI_vals, 30, where=(RSI <= 30), facecolor='green', edgecolor='green')
	# 	ax1.text(0.6, 0.9, '>70 = overbought', va='top', transform=ax1.transAxes, fontsize=textsize)
	# 	ax1.text(0.6, 0.1, '<30 = oversold', transform=ax1.transAxes, fontsize=textsize)
	# 	ax1.set_ylim(0, 100)
	# 	ax1.set_yticks([30, 70])
	# 	ax1.text(0.025, 0.95, 'RSI (%s)' % q, va='top', transform=ax1.transAxes, fontsize=textsize)
	# 	ax1.set_title('{0} daily'.format(stock))


def get_currency_list():
	currency_list = ['CURRFX/MXNUSD.1', 'CURRFX/USDCAD.1', 'CURRFX/NZDUSD.1', 'CURRFX/USDHKD.1', 'CURRFX/USDJPY.1', 'CURRFX/USDSGD.1', 'CURRFX/GBPUSD.1', 'CURRFX/USDZAR.1', 'CURRFX/AUDUSD.1', 'CURRFX/EURUSD.1']
	return currency_list

def get_currency_data(currency_list, num_days, api_key):
	# Calculate dates
	end_date = datetime.date.today()
	start_date = end_date - timedelta(num_days)

	# Initialize data table
	data_table = None
	# Run through currencies, first assignment is initialized
	# Anything past first currency is joined into table
	for currency in currency_list:
		current_column = qdl.get(currency, start_date= start_date, end_date= end_date, authtoken= api_key)
		if data_table is None:
			data_table = current_column
		else:
			data_table = data_table.join(current_column, how= 'left', rsuffix= ' ')

	data_table.columns = ['MXN/USD', 'USD/CAD', 'NZD/USD', 'USD/HKD', 'USD/JPY', 'USD/SGD', 'GBP/USD', 'USD/ZAR', 'AUD/USD', 'EUR/USD']

	return data_table 

def RSI_Calc(currency_data, q):
	delta = currency_data.diff()
	dUp, dDown = delta.copy(), delta.copy()
	dUp[dUp < 0] = 0
	dDown[dDown > 0] = 0

	RolUp = dUp.rolling(window= q, center= False).mean()
	RolDown = dDown.rolling(window = q, center= False).mean().abs()

	RS = RolUp / RolDown
	RSI = 100.0 - (100.0 / (1.0 + RS))
	RSI = RSI.dropna()
	return RSI

if __name__ == "__main__":
	main()
