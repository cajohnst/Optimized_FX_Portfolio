	import matplotlib.pyplot as plt


	'''***call function to download weights_table from google spreadsheeets here***
		weights table must be in same format as returns_table

		***call function to import returns_table with leverage here***
	'''


	#Charts

	#The first chart will show past distributions up to the current period
	#To prevent cramped figure, only plotting last 10 periods
	distribution_chart = weights_table.plot(kind='bar',stacked=True)
	plt.ylim([0,1])
	plt.xlabel('Date')
	plt.ylabel('Distribution')
	plt.title('Distribution vs. Time')
	distribution_chart.legend(loc='center left', bbox_to_anchor=(1, 0.5) , prop= {'size':10})


	#weighted returns are calculated using element-wise multiplication of the weights table and returns table
	weighted_returns = weights_table * returns_table
	print 'weighted returns'
	print weighted_returns
	#portfolio returns are the sum of daily weighted returns.  We will plot portfolio returns
	portfolio_daily_returns= np.sum(weighted_returns.T)
	print 'portfolio sums'
	print portfolio_daily_returns

	plt.figure()
	returns_plot = portfolio_daily_returns.plot()
	plt.xlabel('Date')
	plt.ylabel('Portolio Returns')
	plt.title('Portfolio Returns vs. Time')
	returns_plot.legend(loc= 'upper left' , prop={'size':10})

	plt.show()