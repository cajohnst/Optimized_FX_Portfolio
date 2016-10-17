import requests
import datetime
from datetime import date, timedelta
import pandas as pd 
import settings as sv 
# Hardcoded currencies, to be changed

def main():
	today = sv.end_date 
	today = today.strftime("%Y%m%d")

	'''fxstreet_scraper will be run twice per day, the first time will allow for the user to input predictions 
	for data releases, which will be appended to the Full Event Calendar and used as predictions in the regression 
	estimates.  delete_predictions will delete these predictions from the Event Calendar and replace them with the 
	actual data releases, allowing the Calendar to be correct once more on the days second run through.

	'''

	csv_data = get_csv(today)
	return csv_data

def get_csv(today):
	headers = {
	'Host': 'calendar.fxstreet.com',
	'Connection': 'keep-alive',
	'Pragma': 'no-cache',
	'Cache-Control': 'no-cache',
	'Upgrade-Insecure-Requests': '1',
	'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36',
	'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
	'Referer': 'http://www.fxstreet.com/economic-calendar',
	'Accept-Encoding': 'gzip, deflate, sdch, br',
	'Accept-Language': 'en-US,en;q=0.8,zh-TW;q=0.6'
	}
	url = "https://calendar.fxstreet.com/eventdate/?f=csv&v=2&timezone=UTC&rows=&view=range&start={0}&end={1}&countrycode=AU%2CCA%2CCN%2CEMU%2CFR%2CDE%2CGR%2CIT%2CJP%2CNZ%2CPT%2CES%2CCH%2CUK%2CUS&volatility=2&culture=en&columns=CountryCurrency%2CCountdown".format(today, today)
	
	# url = "https://calendar.fxstreet.com/eventdate/?f=csv&v=2&timezone=UTC&rows=&view=range&start=20160926&end=20160928&countrycode=AU%2CCA%2CCN%2CEMU%2CFR%2CDE%2CGR%2CIT%2CJP%2CNZ%2CPT%2CES%2CCH%2CUK%2CUS&volatility=2&culture=en&columns=CountryCurrency%2CCountdown"
	csv_data = requests.get(url, headers= headers)
	csv_encode = csv_data.text.encode('utf-8')
	
	# with open('event_calendar_today.csv', 'w') as csv_file:
	# 	csv_file.write(csv_encode)

	return csv_encode


if __name__ == "__main__":
	main()
