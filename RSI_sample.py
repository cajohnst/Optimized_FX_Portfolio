import settings as sv 
import pandas as pd 
from pandas import DataFrame
import quandl as qdl
import datetime
import numpy as np
from datetime import date, timedelta
import matplotlib.colors as colors
import matplotlib.finance as finance
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import Pull_Data
from matplotlib.backends.backend_pdf import PdfPages

#Word of caution, it seems quandl data is imperfect for high and low readings before July, 2016

def main():
	#List of currencies to pull data for
	currency_list = sv.get_currency_list()
	currency_quandl_list = sv.get_currency_quandl_list()
	#Create new lists to pull daily lows and highs for the stochastic oscillator
	list_high = [high.replace('1', '2') for high in currency_quandl_list]
	list_low = [low.replace('1', '3') for low in currency_quandl_list]

	max_lag = max(sv.q, sv.nslow, sv.nfast, sv.nema, sv.ma_slow, sv.ma_fast, sv.n, sv.d)

	#Pull this many days of data to return the amount of data user has requested
	pull_data_days = sv.num_days_charts + max_lag

	#Pull data from quandl
	currency_table = Pull_Data.get_currency_data(currency_list, currency_quandl_list, pull_data_days, sv.end_date , sv.auth_tok)
	#Get daily lows from quandl for stochastic oscillator
	low_table = Pull_Data.get_currency_data(currency_list, list_low, pull_data_days, sv.end_date , sv.auth_tok)
	#Get daily highs from quandl for stochastic oscillator
	high_table = Pull_Data.get_currency_data(currency_list, list_high, pull_data_days, sv.end_date , sv.auth_tok)

	# #Calculate RSI for all currency pairs in currency_table
	RSI = RSI_Calc(currency_table, sv.q)

	# #Calculate simple moving averages
	ma_f = moving_average(currency_table, sv.ma_fast, type='simple')
	ma_s = moving_average(currency_table, sv.ma_slow, type='simple')

	#Calculate exponentially weighted moving averages and MACD
	emaslow, emafast, macd = get_MACD(currency_table, nslow= sv.nslow, nfast = sv.nfast)
	ema9 = moving_average(macd, sv.nema, type = 'exponential')

	#Calculate stochastics
	fast_stochastic, slow_stochastic = get_stochastic(currency_table, low_table, high_table, sv.n, sv.d)

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

	daily_charts_pdf = PdfPages('Daily_Charts.pdf')
	'''These plots will eventually be moved to produce_charts'''
	for currency in RSI:
		fig = plt.figure(facecolor='white')
		plt.rc('axes', grid=True)
		plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)

		textsize = 9
		left, width = 0.1, 0.8
		rect1 = [left, 0.6, width, 0.3]
		rect2 = [left, 0.45, width, 0.15]
		rect3 = [left, 0.3, width, 0.15]
		rect4 = [left, 0.15, width, 0.15]
		axescolor = '#f6f6f6'  # the axes background color

		ax1 = fig.add_axes(rect1, axisbg=axescolor)  # left, bottom, width, height
		ax2 = fig.add_axes(rect2, axisbg=axescolor, sharex=ax1)
		ax3 = fig.add_axes(rect3, axisbg=axescolor, sharex=ax1)
		ax4 = fig.add_axes(rect4, axisbg=axescolor, sharex=ax1)

		ax2.plot(RSI.index, RSI[currency], color='blue')
		ax2.axhline(sv.Overbought, color='red')
		ax2.axhline(sv.Oversold, color='green')
		ax2.text(0.6, 0.9, '>70 = overbought', va='top', transform=ax2.transAxes, fontsize=textsize)
		ax2.text(0.6, 0.1, '<30 = oversold', transform=ax2.transAxes, fontsize=textsize)
		ax2.set_ylim(0, 100)
		ax2.set_yticks([30, 70])
		ax2.text(0.025, 0.95, 'RSI (%s)' % sv.q, va='top', transform=ax2.transAxes, fontsize=textsize)
		ax1.set_title('{0} Daily Chart'.format(currency))

		line = ax1.plot(currency_table.index, currency_table[currency], color='black', label='_nolegend_')

		linema20, = ax1.plot(ma_f.index, ma_f[currency], color='blue', lw=2, label='MA (%s)' % sv.ma_fast)
		linema200 = ax1.plot(ma_s.index, ma_s[currency], color='red', lw=2, label='MA (%s)' % sv.ma_slow)

		ax3.plot(macd.index, macd[currency], color='black', lw=2)
		ax3.plot(ema9.index, ema9[currency], color='blue', lw=1)
		ax3.fill_between(macd.index, macd[currency] - ema9[currency], 0, alpha=0.5, facecolor='red', edgecolor= 'maroon')
		ax3.text(0.025, 0.95, 'MACD (%d, %d, %d)' % (sv.nfast, sv.nslow, sv.nema), va='top', transform=ax3.transAxes, fontsize=textsize)

		ax4.plot(fast_stochastic.index, fast_stochastic[currency], color='blue')
		ax4.plot(slow_stochastic.index, slow_stochastic[currency], color= 'yellow')
		ax4.axhline(sv.Overbought_S, color='red')
		ax4.axhline(sv.Oversold_S, color='green')
		ax4.text(0.6, 0.9, '>80 = overbought', va='top', transform=ax4.transAxes, fontsize=textsize)
		ax4.text(0.6, 0.1, '<20 = oversold', transform=ax4.transAxes, fontsize=textsize)
		ax4.set_ylim(0, 100)
		ax4.set_yticks([20, 80])
		ax4.text(0.025, 0.95, 'Stochastic (%s)' % sv.n, va='top', transform=ax4.transAxes, fontsize=textsize)
		# turn off upper axis tick labels, rotate the lower ones, etc
		for ax in ax1, ax2, ax3, ax4:
			if ax != ax4:
				for label in ax.get_xticklabels():
					label.set_visible(False)
		else:
			for label in ax.get_xticklabels():
				label.set_rotation(30)
				label.set_horizontalalignment('right')

		ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
		plt.savefig(daily_charts_pdf, format= 'pdf')
		plt.close(fig)
	daily_charts_pdf.close()


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


if __name__ == "__main__":
	main()
