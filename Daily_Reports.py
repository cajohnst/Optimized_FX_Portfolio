# from __future__ import print_function
import Pull_Data
import fxstreet_scraper
import Optimize_FX_Portfolio
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

	'''
def main():
######################################################################################################################
#Pull and Update data	
	# fxstreet_scraper.main()
#Pull Today's Data
# 	today = datetime.date.today().strftime("%Y%m%d")
# 	fxstreet_scraper.get_csv(today)
# #Begin User Input for Predictions of events
# 	econ_calendar_today = pd.read_csv("event_calendar_today.csv")
# 	econ_calendar_today.Consensus.fillna(econ_calendar_today.Previous, inplace = True)
# 	econ_calendar_today = econ_calendar_today.dropna()


# 	for index, row in econ_calendar_today.iterrows():
# 		prediction = raw_input("Prediction for {0} in {1} given the market consensus is {2}.\n Your Prediction:".format(row['Name'], row['Country'], row['Consensus']))
# 		econ_calendar_today.set_value(index, 'Actual', prediction)

# 	return econ_calendar_today 
# #Export updated CSV
# 	econ_calendar_today.to_csv("event_calendar_today.csv", index=False)
# #Merge predictions to full calendar
# 	fxstreet_scraper.merge_csv("event_calendar_today.csv")

	Pull_Data.main()

######################################################################################################################
#Optimize Portfolio before making expected return predictions
	# Optimize_FX_Portfolio.main()



######################################################################################################################
#Import the merged event/technical tables for each currency pair for prediction
	from Pull_Data import return_predictions 
	print return_predictions
	# prediction_array = Dataframe(return_predictions)
	# prediction_array.append(-1 * return_predictions)
	# print [prediction_array] 

# pbar = opt.matrix(np.mean(returns, axis=1))

######################################################################################################################
# Charts

	# plt.plot(stds, means, 'o')
	# plt.plot(risks, returns, 'y-o')
	# plt.plot(expected_std, expected_return, 'r*', ms= 16)
	# plt.ylabel('Expected Return')
	# plt.xlabel('Expected Volatility')
	# plt.title('Portfolio Efficient Frontier')
	# plt.show()

if __name__ == "__main__":
	main()










