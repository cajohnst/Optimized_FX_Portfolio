# Optimized_FX_Portfolio
The Optimize FX Portfolio Project -Version 1.0 

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
	
	5. Optimize Portfolio
		We then optimize two "different" portfolios utilizing mean-variance portfolio optimization.  The first portfolio according to the traditional Markowitz approach, using the 
		mean of historical returns as the expected return.  The second uses our predicted returns from the regression process as our "expected" return.  Mean-variance optimization 
		requires an additional minimum return.  We also append our returns with the return of a "risk free rate".  Now, using the library CvxOpt, we solve for the portfolio weights
		which minimize portfolio variance while simultaneously achieving the minimum daily return.  We return the weights for both portfolios, and append them to a Google Spreadsheet
		consisting of historical currency pair weights.  		

	6. Charts
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
