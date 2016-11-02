import Pull_Data
import fxstreet_scraper
import Optimize_FX_Portfolio
import RSI_sample
import rollover_google_sheet
import weights_google_sheet
import settings as sv
import import_spot_test
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
import cvxopt as opt 
from cvxopt import matrix
import matplotlib.colors as colors
import matplotlib.finance as finance
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.mlab as mlab
import matplotlib.font_manager as font_manager
import quandl as qdl 
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm 
import argparse


######################################################################################################################
'''
The Optimize FX Portfolio Project:
	

	Optimize FX Portfolio implements optimization and machine learning techniques in order to efficiently manage a
	portfolio of currencies.  Portfolio Optimiztion, as described by Markowitz, is very accurate at minimizing portfolio variance,
	but assumes the mean as the expected return.  We believe this method is inefficient at capturing returns, and therefore employ
	regression prediction techniques to allow ourselves a better day-to-day return estimate.  The following is the process for 
	completing this task.

	1. Pull Data
		Data required for this project comes from a variety of sources.  We first pull historical daily exchange rates
		from the Quandl repository, as well as federal funds futures and the effective funds rate to calculate the probability
		of a Federal Reserve rate hike, and finally, to pull comparative data in the form of a benchmark asset, which defaults
		to the S&P 500 index.  Additionally, a unique return source to the Foreign Exchange market is the existence of rollover.
		Rollover is the parity in interest rates set between the central banks of differing monetary bases.  Rollover can be positive
		or negative given a traders long or short position, and allows the trader to implement a "carry-trade" strategy.  Since rollover
		can significantly impact strategy, we believe adding rollover to our returns more accurately describes and predicts future returns.
		However, the amount of rollover given is unique to each Forex broker, and may not accurately reflect the actual interest rate parity, therefore,
		we have accumulated real rollover rates from the broker Forex.com for 10 of the most traded currencies and compiled them into a Google 
		Spreadsheet.  This data has been compiled since July 2016.  Finally, we have compiled a spreadsheet of significant (by volatility)
		economic data releases, which we use in our regression function.  We believe that the foreign exchange market can be more accurately 
		traded utilizing both fundamental and technical methods.  We utilize the deviation between the report consensus and actual values to predict
		generated volatility and returns for each currency pair.  The economic calendar dates back to October 2013 and is appended daily. 

	2. Compute technicals
		In addition to the fundamentals, we have chosen to implement some basic technical trading metrics that we believe may also assist in predicting 
		trend reversal.  These chosen metrics widely known to market participants, are RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), 
		and Stochastics.  Once these are calculated, a chart is drafted for each currency pair including the exchange rate and the three mean-reversion techniques 
		plotted below along the same time axis.  This code is currently located in RSI_sample

	3. Predict today's event releases
		User input for predicting today's event releases.  These can be input retroactively once data is released to predict the closing rate for the currency pair
		that day.  Given today's releases are significant (able to move the market rate) and match with the event name in a event dictionary of relevant releases,  
		data for today's release, as well as all releases of this event in the past number of specified days will be pulled and joined together with other events, 
		the technical metrics, and rate hike probabilities into a set of dataframes for each currency pair.  These frames constitute the tables which will be used 
		to fit and predict regression models for each currency pair.  Events which are more relevant to a particular pair will have a higher impact on a pair, the 
		same event may have a low impact on a different pair, but every event release is used in the regression fit for each currency pair.  The technicals are
		individualized for each pair, and US rate hike probabilties are joined as event release data is.  Pull_Data returns a list of these joined tables as dataframes
		for use by the ridge regression function.

	4. Ridge Regression
		Using the ridge regression module in scikit-learn, we take the list of dataframes, using the technical indicators and event releases as regressors and the exchange
		rate as the dependent variable.  We hope to more accurately predict the closing return given today's data releases and technical indicators.  We use ridge regression 
		because we expect that multicollinearity is prevalent in the foreign exchange market, especially since many of the most traded currency pairs are denominated in US
		dollars.  Once we have fit the historical data, we predict returns based on the regressors of today's data.  This includes estimates of technical indicators (using 
		current spot rate at time of run) as well as predictions for today's economic data releases.  The closer to end of trading day, the more accurate the predictions will
		be.  Of course, this in itself is a risk-return tradeoff.  Ideally, one would run updated regression predictions throughout the day after each data release becomes available, 
		or simulate many different data release scenarios.  Making these predictions is currently not an automated process.
	
	5. Calculate returns
		Return calculations not only include the change between currencies over an interval of time, we have also included additions and negations for rollover and execution costs. 
		In calculating actual portfolio returns, we take the sum of intraday returns to calculate returns on the day.  Using these daily returns, we calculate the return over a period
		longer than one day as the cumulative (compounded) return over the specified period.  This is an important assumption, that intraday returns are not compounded.  We have also 
		incorporated our adjusted intraday portfolio return into the minimum return constraint when optimizing the portfolio.  The minimum return is calculated as the minimum return value
		assigned in settings minus the accumulated intraday return.  Our predicted return values also take into account amounts of rollover and execution costs per trade.  Thus, we have 
		more accurately optimized our portfolio by incorporating returns, rollover, and execution costs rather than just returns.

	6. Optimize Portfolio
		We then optimize two "different" portfolios utilizing mean-variance portfolio optimization.  The first portfolio according to the traditional Markowitz approach, using the 
		mean of historical returns as the expected return.  The second uses our predicted returns from the regression process as our "expected" return.  Mean-variance optimization 
		requires an additional minimum return.  We also append our returns with the return of a "risk free rate".  Now, using the library CvxOpt, we solve for the portfolio weights
		which minimize portfolio variance while simultaneously achieving the minimum daily return.  We return the weights for both portfolios, and append them to a Google Spreadsheet
		consisting of historical currency pair weights.  		

	7. Charts
		In addition to the exchange rate charts with technical indicators, we have included some basic charts indicating portfolio performance.  First, a Markowitz
		"Bullet", a chart with simulated portfolios given past returns data, and "efficient frontiers" for both a portfolio optimized on the mean as expected return
		and the predicted returns as the expected return.  The stars indicate the mean-variance optimal solution for each portfolio.  Following is a chart showing the
		change in distribution of currencies in the portfolio according to the predicted returns optimal solution.  Past portfolio weights are taken from a Google Spreadsheet
		and plotted in stacked-bar form over a given interval.  In sequence, the next chart is a potrayal of cumulative returns comparing the actual portfolio returns to 
		a benchmark asset, which defaults to the S&P 500.  After this is a chart depicting the change in rolling Sharpe Ratios, also comparing the portfolio results to those
		of the benchmark asset.  The final set of charts depict a rolling Value at Risk calculation, and a histogram of past portfolio returns to assess normality.  We export 
		these charts to two separate PDF files, the daily exchange rate charts for individual currency pairs in one file, and overall portfolio metrics in the other.

	While the Optimize FX Portfolio project has been intended as a fun side project between friends, for one, this has been a valuable experience in taking beginning steps with 
	the python language, pandas, and other vital open source libraries.  Still, we believe this program to be of some value, but advise caution when investing.
	Risk is inherent in every investment decision, therefore it is important to understand the relationship between risks and expected return prior to making such a decision.
	Thank you for your interest!  Any suggestions or comments can be sent to cajohnst1@gmail.com


	Authors: Kevin Jang and Carter Johnston

	Copyright (c) 2016
'''
######################################################################################################################

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--no-update', action='store_true', help='Use if you do not wish to update weights google sheet.')
	args = parser.parse_args()

	# Flag for updating weights on google sheets
	update_weights = True
	if args.no_update:
		update_weights = False

	np.random.seed(919)
	currency_list = sv.get_currency_list() 
	currency_quandl_list = sv.get_currency_quandl_list() 

#Get currency data and return as percentage returns
	live_rates = import_spot_test.main()
	# Export run-time spot rates to google sheet for calculating portfolio returns
	to_google_sheet = live_rates.values.flatten().tolist()
	weights_google_sheet.main(to_google_sheet, 'Spot-Rates')
	currency_table = Pull_Data.get_currency_data(currency_list, currency_quandl_list, sv.num_days_optimal_portfolio, sv.end_date, sv.auth_tok)
	currency_table = currency_table.append(live_rates)
	returns_table = currency_table.pct_change(periods= sv.shift).dropna()
	returns_table.drop(returns_table.index[:1], inplace=True)

# Create a list of tables of relevant data to form regression estimates on.  
# A unique regression table is created for each currency pair in the currency list
	regression_table = Pull_Data.main()
# Create list of return predictions for each currency pair based on regression tables, convert list to matrix form for cvxopt.
	pred_pbar = predict_returns(regression_table, sv.interest_rate)

# Subtract intraday returns and execution costs from predicted returns 
	intraday_returns = returns_table.iloc[-1].values.flatten().tolist() 
	intraday_returns[:] = [intrd * 100 * sv.leverage for intrd in intraday_returns]
	intraday_with_shorts = []
	for val in intraday_returns:
		short = -val 
		intraday_with_shorts.append(val)
		intraday_with_shorts.append(short)
	execution_returns = [float(exec_costs)/ rates for exec_costs, rates in zip(sv.execution_costs, live_rates.values.flatten().tolist())]
	execution_returns[:] = [exec_returns * 100 * sv.leverage for exec_returns in execution_returns]
	execution_with_shorts = []
	for cost in execution_returns:
		short = cost 
		execution_with_shorts.append(cost)
		execution_with_shorts.append(short)
	pred_pbar[:] = [prediction - intraday - exec_costs for prediction, intraday, exec_costs in zip(pred_pbar, intraday_with_shorts, execution_with_shorts)]

# Import rollover table to add to regression return predictions
	rollover_table = rollover_google_sheet.pull_data(sv.num_days_optimal_portfolio, currency_list)
	rollover_table = rollover_table / 100
	merge_table = Optimize_FX_Portfolio.merge_tables(returns_table, rollover_table)
	merge_table = 100 * sv.leverage * merge_table.dropna()
	merge_table['RF'] = sv.interest_rate
	return_vector = (np.asarray(merge_table).T) 
	mean_rollover = np.mean(rollover_table, axis=0)

	# Calculate daily returns as the cumulative sums of intraday moves and add risk free return to calculate risk metrics and cumulative returns charts
	actual_returns = weights_google_sheet.pull_data(sv.num_days_optimal_portfolio, 'Spot-Rates')
	actual_returns.index = actual_returns.index.date
	actual_returns = actual_returns.pct_change(periods= sv.shift).dropna()
	actual_returns = sv.leverage * 100 * actual_returns
	# actual_returns = actual_returns.groupby([actual_returns.index]).sum()
	#add rollover 
	actual_returns['RF'] = sv.interest_rate * 100

	# Import weights tables for portfolio risk metrics 
	historical_weights = weights_google_sheet.pull_data(sv.num_days_optimal_portfolio, 'Prediction')
	historical_weights.index = historical_weights.index.date

	actual_returns = rollover_to_returns(historical_weights, actual_returns, rollover_table)

	#execution costs is a list in settings.py 
	#spot_rates is a series flattened to a list representing the current rate
	execution_returns = [float(exec_costs)/ rates * -100 * sv.leverage for exec_costs, rates in zip(sv.execution_costs, live_rates.values.flatten().tolist())]

	# Subtract execution costs from returns
	# If change in weight value does not surpass weight threshold, no execution is made, therefore, weights revert to previous value
	historical_weights, actual_returns = execution_to_returns(historical_weights, actual_returns, execution_returns)

	# Calculate portfolio returns by multiplying currency weights and returns, summing intraday returns into daily returns
	portfolio_returns = historical_weights.multiply(actual_returns, axis= 0)
	portfolio_returns = portfolio_returns.dropna()
	portfolio_returns = portfolio_returns.groupby([portfolio_returns.index]).sum()
	portfolio_returns['Portfolio Returns'] = portfolio_returns.sum(axis = 1)

	# For comparison, import a benchmark asset to compare portfolio sharpe ratios over time as well as VaR analysis
	benchmark_list = ['SPY']
	benchmark_quandl_list = ['GOOG/NYSE_SPY.4']

	benchmark = Pull_Data.get_benchmark(benchmark_list, benchmark_quandl_list, sv.num_days_optimal_portfolio, sv.end_date, sv.auth_tok, sv.shift)
	benchmark['Benchmark'] = benchmark.sum(axis = 1)
	benchmark_for_VaR = benchmark['Benchmark']

	# Calculate benchmark rolling sharpe ratios
	benchmark_sharpe = calc_sharpe(benchmark, sv.interest_rate, sv.rolling_period)
	#Calculate portfolio rolling sharpe ratio
	portfolio_sharpe = calc_sharpe(portfolio_returns['Portfolio Returns'], sv.interest_rate, sv.rolling_period)

	# Create pandas dataframe with benchmark and portfolio rolling sharpe ratios
	sharpe_df = benchmark_sharpe.join(portfolio_sharpe, how='left', rsuffix='')
	sharpe_df.dropna(inplace= True)

	total_return = portfolio_returns['Portfolio Returns']
	# Adjust minimum return for accumulated intraday returns
	if datetime.date.today() in total_return.index.values:
		minimum_return = sv.rminimum - total_return.iloc[-1]
	else:
		minimum_return = sv.rminimum

	# if minimum_return < 0:
	# 	pred_pbar = [prediction * sv.reduced_leverage / sv.leverage for prediction in pred_pbar]
	# 	mean_rollover = [rollover * sv.reduced_leverage / sv.leverage for rollover in mean_rollover]

	mean_rollover = sv.leverage * opt.matrix(np.append(mean_rollover, np.array(0)))
	pred_pbar.append(sv.interest_rate)
	pred_pbar = opt.matrix(np.asarray(pred_pbar))
	# Final prediction to be used in Optimize_FX_Portfolio is the summation of the regression prediction and the average rollover 
	# computed over the rollover period.
	pbar = pred_pbar + mean_rollover
	#Create pandas dataframe with benchmark and portfolio cumulative returns
	cumulative_returns_df = (benchmark/100).join(total_return/100, how='left', rsuffix ='')
	cumulative_returns_df.dropna(inplace= True)
	cum_return_func = lambda x: ((1 + x).cumprod() - 1) * 100
	cumulative_returns_df = cumulative_returns_df.apply(cum_return_func, axis=0)

	#Calculate value at risk estimates over the rolling period (default is 95% confidence)
	var, mean, std= calc_VaR(total_return, sv.portfolio_value, sv.rolling_period, confidence_level= 0.95)
	var_benchmark, mean_benchmark, std_benchmark= calc_VaR(benchmark_for_VaR, sv.portfolio_value, sv.rolling_period, confidence_level= 0.95)

	#mu and standard deviation for computing fitted z-score
	#num_bins for number of bins in histogram
	mu = mean.iloc[-1]
	sigma = std.iloc[-1] 

	#Create pandas dataframe with Value at Risk elements
	VaR_df = pd.concat([var_benchmark, var], axis = 1)
	VaR_df.dropna(inplace= True)
	VaR_df.columns = ['Benchmark VaR', 'Portfolio VaR']

########################################################################################################################
# Optimize portfolio weights based on return predictions.
# For comparison, first calculate the optimized portfolio based purely on historical data
	weights, expected_return, expected_std, risks, returns, means, stds = Optimize_FX_Portfolio.main()
# Now create an optimized portfolio and an "efficient frontier" for the predicted returns portfolio
	pred_weights, pred_return, pred_std = Optimize_FX_Portfolio.OptimalWeights(return_vector, minimum_return, pbar)
	predicted_risks, predicted_returns = Optimize_FX_Portfolio.EfficientFrontier(return_vector, pbar)

# Export the optimized portfolio weights to google sheet
# Optimize_FX_Portfolio returns weights of both long and short positions as well as a risk free rate.
# These weights are condensed to net-long positions in the google sheet since having simultaneous long and short positions is impractical.
	currency_list.append('Risk Free')
	condensed_weights = consolidate_weights(weights)
	condensed_pred_weights = consolidate_weights(pred_weights)
# Export weights to google sheet
	if update_weights:
		weights_google_sheet.main(condensed_weights, 'Mean-Variance')
		weights_google_sheet.main(condensed_pred_weights, 'Prediction')
	
# ######################################################################################################################
# # Charts
	#Charts including Price, RSI, MACD, and Stochastics for each currency pair
	#These charts will be exported to a separate PDF document than portfolio risk metrics
	RSI_sample.main()

	# Define PDF to append charts
	daily_report_pdf = PdfPages('Daily_Report.pdf')

	#Chart displaying the efficient frontier, 5000 random portfolios, and stars for minimum variance given a minimum return
	#The red star represents a mean-variance portfolio given historical returns, and the green star represents a mean-variance
	#Portfolio accounting for technical and fundamental daily analysis predictions.
	plt.plot(stds, means, 'o')
	plt.plot(risks, returns, 'y-o')
	plt.plot(predicted_risks, predicted_returns, '-o', color= 'orange')
	plt.plot(expected_std, expected_return, 'r*', ms= 16)
	plt.plot(pred_std, pred_return, 'g*', ms = 16)
	plt.ylabel('Expected Return')
	plt.xlabel('Expected Volatility')
	plt.title('Portfolio Efficient Frontier')
	plt.savefig(daily_report_pdf, format= 'pdf')
	
	# #Chart which displays the change in the distribution of weights in the portfolio over the last 10 time intervals as defined in main()
	legend_cols = int(len(historical_weights.columns)/ 3)
	distribution_chart = historical_weights.iloc[-10 * sv.distribution_interval::sv.distribution_interval, :].plot(kind='bar',stacked=True, colormap= 'Paired')
	plt.ylim([-1,1])
	plt.xlabel('Date')
	plt.ylabel('Distribution')
	plt.title('Distribution vs. Time')
	plt.tight_layout()
	plt.legend(loc= 'upper left', prop= {'size': 10}, ncol= legend_cols)
	plt.savefig(daily_report_pdf, format = 'pdf')

	#Plot cumulative returns over given period
	cumulative_returns_plot = cumulative_returns_df.plot()
	plt.xlabel('Date')
	plt.ylabel('Cumulative Returns (%)')
	plt.title('Portfolio Returns vs. Benchmark')
	cumulative_returns_plot.legend(loc= 'upper left', prop={'size':10})
	plt.savefig(daily_report_pdf, format='pdf')

	#Plot sharpe ratios
	sharpe_plot = sharpe_df.plot()
	plt.xlabel('Date')
	plt.ylabel('Rolling ({0})-Day Sharpe Ratio'.format(sv.rolling_period))
	plt.title('Portfolio Sharpe Ratio vs. Benchmark')
	sharpe_plot.legend(loc= 'upper left', prop={'size':10})
	plt.savefig(daily_report_pdf, format='pdf')

	#Create VaR chart
	VaR_plot = VaR_df.plot()
	plt.xlabel('Date')
	plt.ylabel('Rolling ({0}) day Value at Risk Per ${1}'.format(sv.rolling_period, sv.portfolio_value))
	plt.title('Daily Value at Risk Per ${0} for Portfolio and {1}'.format(sv.portfolio_value, benchmark_list[0]))
	VaR_plot.legend(loc= 'upper left', prop={'size':8})
	plt.savefig(daily_report_pdf, format='pdf')

	plt.clf()

	#Create portolio returns histogram to assess normality
	n, bins, patches = plt.hist(total_return, bins= sv.num_bins, normed=1, facecolor= 'green')
	y = mlab.normpdf(bins, mu, sigma)
	l = plt.plot(bins, y, 'r--', linewidth=1)
	plt.xlabel('Return Percentage')
	plt.ylabel('Probability')
	plt.title('Portfolio Return Distribution ({0} Bins)'.format(sv.num_bins))
	plt.grid(True)
	plt.savefig(daily_report_pdf, format='pdf')

	# End pdf report
	daily_report_pdf.close()


######################################################################################################################
#Predict Returns based on Ridge Regression Estimates
def predict_returns(regression_table, interest_rate):
	# Compute a ridge regression given the table of unique events data and technicals.
	# Use ridge regression since it is very likely that changes in the foreign exchange market are multicollinear.
	# Issues: R-squared values cannot be calculated if prediction is for a single row of data (predict today's returns as we would like to)
	#		  Each daily regression is unique, therefore it is difficult to test accuracy without simulation.
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
	# short_list.append(interest_rate)
	return short_list 

def calc_sharpe(return_array, interest_rate, rolling_period):
	#Calculate Sharpe Ratio to compare Benchmark Sharpe to Portfolio Sharpe

	mean_return = RSI_sample.moving_average(return_array, rolling_period, type = 'simple')

	std_return = return_array.rolling(window = rolling_period).std()

	sharpe = (mean_return - interest_rate)/ std_return 
	sharpe = sharpe.dropna()

	return sharpe 


def calc_VaR(return_array, initial_value, rolling_period, confidence_level = 0.95):

	#Value at Risk is calculated as the initial portfolio value multiplied by both the standard deviation
	#and the z-score value given the mean and standard deviation of portfolio returns.

	mean_return = RSI_sample.moving_average(return_array, rolling_period, type= 'simple')
	mean_return.name = "Mean"

	std_return = return_array.rolling(window= rolling_period).std()
	std_return.name = "STD"

	alpha = norm.ppf(1- confidence_level, mean_return, std_return)
	alpha_series = pd.Series(alpha, index= mean_return.index, name = 'alpha')

	VaR = pd.concat([mean_return, std_return, alpha_series], axis = 1)
	VaR['VaR'] = -(initial_value * (VaR['STD'] / 100) * VaR['alpha'])
	VaR = VaR['VaR']

	return VaR, mean_return, std_return 


def consolidate_weights(weights_array):

	# Weights are returned from Optimize_FX_Portfolio as both long and short positions.  Since this is not
	# logical, consolidate the weights as long - short position for true currency 'weight'.
	rf = weights_array.pop()
	num_weights= len(weights_array)/ 2

	weights_vector = [weights_array[2*p+1]-weights_array[2*p] for p in range(num_weights)]
	weights_vector.append(rf)

	return weights_vector

def rollover_to_returns(weights_table, returns_table, rollover_table):
	for row_index, current_date in enumerate(weights_table.index):
		if row_index > 0:
			date_delta = (current_date - weights_table.index[row_index-1]).days
			if date_delta > 0:
				for column_index, value in weights_table.iloc[row_index].iteritems():
					if column_index != 'RF':
						if value > 0:
							returns_table.iloc[row_index, returns_table.columns.get_loc(column_index)] += rollover_table.loc[current_date][column_index+' - L'] * date_delta * sv.leverage
						elif value < 0:
							returns_table.iloc[row_index, returns_table.columns.get_loc(column_index)] += rollover_table.loc[current_date][column_index+' - S'] * date_delta * sv.leverage 
						else:
							continue
	return returns_table

def execution_to_returns(weights_table, returns_table, execution_list):
	n_weights = len(weights_table)
	# make temporary copy of historical weights
	# Do not want to change while iterating
	tmp_weights = weights_table.copy(deep= True)
	tmp_weights.drop(tmp_weights.index[[0]])
	# row_index contains the index in integer format
	# row contains the index in datetime format
	for row_index, row in enumerate(weights_table.index):
		# Skip first row
		if row_index > 0:
			# column_index contains currency name, ex: USD/MXN
			for column_index, boolean in  (abs(tmp_weights.iloc[row_index] - tmp_weights.iloc[row_index-1]) >= sv.weight_threshold).iteritems():
				if column_index != 'RF':
					if boolean:
						returns_table.iloc[row_index, weights_table.columns.get_loc(column_index)] += execution_list[tmp_weights.columns.get_loc(column_index)]
						if row_index == n_weights - 1:
							#Print a better message than this
							print 'Currency: {0}, New Weight Allocation: {1}'.format(column_index, tmp_weights.iloc[row_index, weights_table.columns.get_loc(column_index)])
					else:
						tmp_weights.iloc[row_index] = tmp_weights.iloc[row_index-1]
	return weights_table, returns_table


if __name__ == "__main__":
	main()
