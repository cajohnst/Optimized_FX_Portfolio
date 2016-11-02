
import datetime
from datetime import timedelta, date
import csv
import requests
import pandas as pd 
import StringIO
import weights_google_sheet
import Pull_Data
import settings as sv 

def main():

	currency_list = sv.get_currency_list()

	yahoo_list = ['MXN=X', 'USDCAD=X', 'NZDUSD=X', 'HKD=X', 'USDJPY=X', 'USDSGD=X', 'GBPUSD=X', 'USDZAR=X', 'AUDUSD=X', 'EURUSD=X']
	live_quotes = None 
	for ticker in yahoo_list:
		url = 'http://chartapi.finance.yahoo.com/instrument/1.0/{0}/chartdata;type=quote;range=1d/csv'.format(ticker)

		with requests.Session() as s:
			download = s.get(url)

			decoded_content = download.content.decode('utf-8')

			csv_buffer = StringIO.StringIO(decoded_content)
			yahoo_finance_data = pd.read_csv(csv_buffer, skiprows= 18, header= None, index_col= False)
			yahoo_finance_data[0] = pd.to_datetime(yahoo_finance_data[0], unit= 's')
			yahoo_finance_data[0] = yahoo_finance_data[0].apply(lambda x:x.date())
			yahoo_finance_data[0] = pd.to_datetime(yahoo_finance_data[0])
			yahoo_finance_data = yahoo_finance_data.iloc[-1][:2]

			live_quote = pd.DataFrame(yahoo_finance_data)
			live_quote = live_quote.transpose()
			live_quote = live_quote.set_index(0)

		if live_quotes is None:
			live_quotes = live_quote
		else:
			live_quotes = live_quotes.join(live_quote, how= 'left', rsuffix= ' ')
	live_quotes.columns = currency_list 
	live_quotes.index.names = ['DateTime']

	return live_quotes


if __name__ == "__main__":
	main()