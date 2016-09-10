''' 1. Pull_data
	2. Run RSI_sample, MACD_sample, Futures_vs_Spot, and Events
	3. Run regression models for different event scenarios along with the rest of the info
	4. Use regression model to adjust pbar
	5. Run Optimize_FX_Portfolio with adjusted pbar
	6. Produce_charts showing regression models for each currency in currency list, models for days events, efficient frontier,
		and finally portfolio statistics (show portfolio distribution, charts for each holding (with RSI and MACD), and chart of portfolio vs. various metrics)
	7. Save as a markdown or PDF
	8. Automate run to Heroku in early am