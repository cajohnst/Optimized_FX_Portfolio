import requests
import datetime
from datetime import date, timedelta
# Hardcoded currencies, to be changed

def main():
	today = datetime.date.today().strftime("%Y%m%d")
	return get_csv(today)

def get_csv(input_date):
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
	url = "https://calendar.fxstreet.com/eventdate/?f=csv&v=2&timezone=UTC&rows=&view=range&start={0}&end={1}&countrycode=AU%2CCA%2CCN%2CEMU%2CFR%2CDE%2CGR%2CIT%2CJP%2CNZ%2CPT%2CES%2CCH%2CUK%2CUS&volatility=0&culture=en&columns=CountryCurrency%2CCountdown".format(input_date, input_date)
	csv_data = requests.get(url, headers=headers)
	csv_encode = csv_data.text.encode('utf-8')
	
	with open('event_calendar_today.csv', 'w') as csv_file:
		for row in csv_encode.split('/r/n'):
			csv_file.write(row)

	return csv_data

if __name__ == "__main__":
	main()
