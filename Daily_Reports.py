# from __future__ import print_function
import Pull_Data
import fxstreet_scraper
import Optimize_FX_Portfolio
import RSI_sample
import matplotlib.pyplot as plt 
import sklearn as sklearn
from sklearn import preprocessing
from sklearn.linear_model import Ridge
import numpy as np 
import datetime
from datetime import timedelta, date
from cvxopt import matrix
import pandas as pd 
from pandas import Series, DataFrame
import rollover_google_sheet
import cvxopt as opt 
from cvxopt import matrix
import matplotlib.colors as colors
import matplotlib.finance as finance
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.mlab as mlab
import matplotlib.font_manager as font_manager
import quandl as qdl 
import weights_google_sheet

''' 1. Pull_data ***If num_days > 100, drop Stochastic column ***STATUS: Complete ***
	2. Run RSI_sample, MACD_sample, Futures_vs_Spot, and Events ***STATUS: Complete ***
	3. Run regression models for different event scenarios along with the rest of the info
	4. Use regression model to adjust pbar
	5. Run Optimize_FX_Portfolio with adjusted pbar
	6. Produce_charts showing regression models for each currency in currency list, models for days events, efficient frontier,
		and finally portfolio statistics (show portfolio distribution, charts for each holding (with RSI and MACD), and chart of portfolio vs. various metrics)
	7. Save as a markdown or PDF
	8. Automate run to Heroku of fxstreet_scraper.main(today) at 11:50pm
	9. weights table(2 sheets), event calendars, and overall markdown file to google drive..

	Print warning if Name in event_calendar_today is not found in Econ Dict

	'''
def main():
######################################################################################################################
# 	auth_tok = "kz_8e2T7QchJBQ8z_VSi"
# 	end_date = datetime.date.today()
# 	np.random.seed(919)
	currency_list = Pull_Data.get_currency_list()
# 	currency_quandl_list = Pull_Data.get_currency_quandl_list()
# 	num_days = 200
# # 	# rollover_days = 50
# # 	#Compute returns with shift percentage change delay (daily = 1)
# 	shift = 1
# # 	#Compute returns
# 	currency_table = Pull_Data.get_currency_data(currency_list, currency_quandl_list, num_days, end_date, auth_tok)
# 	returns_table = currency_table.pct_change(periods= shift).dropna()
# 	returns_table.drop(returns_table.index[:1], inplace=True)
# 	interest_rate_prediction = 2/ float(365)
# 	regression_table = Pull_Data.main()
# 	pred_pbar = predict_returns(regression_table, interest_rate_prediction)
# 	pred_pbar = opt.matrix(np.asarray(pred_pbar))

# 	rminimum = 100/float(252)

# 	# # Input Leverage
# 	leverage = 10

# 	rollover_table = rollover_google_sheet.pull_data(num_days)

# 	rollover_table = rollover_table / 100
# 	merge_table = Optimize_FX_Portfolio.merge_tables(returns_table, rollover_table)
# 	merge_table = 100 * leverage * merge_table.dropna()
# 	merge_table['RF'] = interest_rate_prediction

# 	return_vector = (np.asarray(merge_table).T) 

# 	mean_rollover = np.mean(rollover_table, axis=0)
# 	mean_rollover = leverage * opt.matrix(np.append(mean_rollover, np.array(0)))

# 	pbar = pred_pbar + mean_rollover
# 	print pbar 

# ######################################################################################################################
# #Optimize Portfolio before making expected return predictions
	weights, expected_return, expected_std, risks, returns, means, stds = Optimize_FX_Portfolio.main()
	# pred_weights, pred_return, pred_std = Optimize_FX_Portfolio.OptimalWeights(return_vector, rminimum, pbar)
	# predicted_risks, predicted_returns = Optimize_FX_Portfolio.EfficientFrontier(return_vector, pbar)
	currency_list.append('Risk Free')
	condensed_weights = consolidate_weights(weights, currency_list)

	weights_google_sheet.main(condensed_weights, 'Prediction')
	weights_google_sheet.main(condensed_weights, 'Actual')

	print condensed_weights
	
#  # Save weights in separate sheets or files index by date, columns not as rollover_table columns.  Example: Column: EUR/USD : value = [(EUR/USD -L) - (EUR/USD -S)], RF weight remains as RF.

# ######################################################################################################################
# # Charts
	#Charts including Price, RSI, MACD, and Stochastics for each currency pair
	# RSI_sample.main()

	#Chart displaying the efficient frontier, 5000 random portfolios, and stars for minimum variance given a minimum return
	#The red star represents a mean-variance portfolio given historical returns, where the green star represents a mean-variance
	#Portfolio accounting for technical and fundamental daily analysis predictions.
	# plt.plot(stds, means, 'o')
	# plt.plot(risks, returns, 'y-o')
	# plt.plot(predicted_risks, predicted_returns, '-o', color= 'orange')
	# plt.plot(expected_std, expected_return, 'r*', ms= 16)
	# plt.plot(pred_std, pred_return, 'g*', ms = 16)
	# plt.ylabel('Expected Return')
	# plt.xlabel('Expected Volatility')
	# plt.title('Portfolio Efficient Frontier')
	# plt.show()

	''' The following charts require the existence of a weights table and will be uncommented when the weights table is completed. '''

	# distribution_chart = weights_table.plot(kind='bar',stacked=True)
	# plt.ylim([0,1])
	# plt.xlabel('Date')
	# plt.ylabel('Distribution')
	# plt.title('Distribution vs. Time')
	# distribution_chart.legend(loc='center left', bbox_to_anchor=(1, 0.5) , prop= {'size':10})
	

	# benchmark_list = ['SPY']
	# benchmark_quandl_list = ['GOOG/NYSE_SPY.4']

	# benchmark = get_benchmark(benchmark_list, benchmark_quandl_list, num_days, end_date, auth_tok, shift)
	# benchmark_sharpe = calc_sharpe(benchmark, interest_rate_prediction)

	# portfolio_returns = weights * returns_table
	# portfolio_sharpe = calc_sharpe(portfolio_returns, interest_rate_prediction)


	# #weighted returns are calculated using element-wise multiplication of the weights table and returns table

	# #portfolio returns are the sum of daily weighted returns.  We will plot portfolio returns
	# portfolio_cum_sum= np.sum(portfolio_returns.T)    
	# benchmark_cum_sum = np.sum(benchmark.T)

	# plt.figure()
	# returns_plot = portfolio_cum_sum.plot()
	# benchmark_plot = benchmark_cum_sum.plot()
	# plt.xlabel('Date')
	# plt.ylabel('Cumulative Returns')
	# plt.title('Portfolio Returns vs. Benchmark')
	# returns_plot.legend(loc= 'upper left' , prop={'size':10})

	# plt.show()

	# plt.figure()
	# portfolio_sharpe_plot = portfiolio_sharpe.plot()
	# benchmark_sharpe_plot = benchmark_sharpe.plot()
	# plt.xlabel('Date')
	# plt.ylabel('Sharpe Ratio')
	# plt.title('Portfolio Sharpe Ratio vs. Benchmark')
	# portfolio_sharpe_plot.legend(loc= 'upper left', prop={'size':10})

	# plt.show()


	# return benchmark_sharpe


######################################################################################################################
#Predict Returns based on Ridge Regression Estimates
def predict_returns(regression_table, interest_rate):
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
		# r_square_to_float = float(clf.score(x_test, y_test))
		# r_squares.append(r_square_to_float)
	short_list = []
	for pred in pred_pbar:
		short = -pred 
		short_list.append(pred)
		short_list.append(short)
	short_list.append(interest_rate)
	return short_list 

def calc_sharpe(return_array, interest_rate):
	#Calculate Sharpe Ratio to compare Benchmark Sharpe to Portfolio Sharpe
	rolling_period = 50

	mean_return = RSI_sample.moving_average(return_array, rolling_period, type = 'simple')

	std_return = return_array.rolling(window = rolling_period).std()

	sharpe = (mean_return - interest_rate)/ std_return 
	sharpe = sharpe.dropna()

	return sharpe 

def get_benchmark(benchmark_list, benchmark_quandl_list, num_days, end_date, api_key, shift):
	start_date = end_date - timedelta(num_days)
	benchmark_table = qdl.get(benchmark_quandl_list, start_date = start_date, end_date = end_date, authtoken= api_key)
	benchmark_table.columns = benchmark_list 
	benchmark_returns = benchmark_table.pct_change(periods= shift).dropna() * 100 
	benchmark_returns.drop(benchmark_returns.index[:1], inplace=True)

	return benchmark_returns

def consolidate_weights(weights_array, column_list):
	rf = weights_array.pop()
	num_weights= len(weights_array)/ 2

	weights_vector = [weights_array[2*p+1]-weights_array[2*p] for p in range(num_weights)]
	weights_vector.append(rf)

	return weights_vector 

if __name__ == "__main__":
	main()
