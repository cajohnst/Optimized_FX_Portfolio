from bs4 import BeautifulSoup
import dryscrape

# Hardcoded currencies, to be changed
currency_list = ['DEXUSEU', 'DEXUSUK', 'DEXUSAL', 'DEXSFUS', 'DEXUSNZ', 'DEXSIUS', 'DEXMXUS', 'DEXJPUS', 'DEXCAUS', 'DEXHKUS']

def generate_rollover(currency_list):
	# Begin session to capture all values, must have JS enabled
	session = dryscrape.Session()
	url = 'http://www.forex.com/uk/trading-platforms/forextrader/pricing/rollovers.html'
	session.visit(url)
	html = session.body()

	# Begin parsing html
	soup = BeautifulSoup(html, 'lxml')

	return_dict = dict()

	# Iterate through each currency to find short and long roll values
	for currency in currency_list:
		try:
			short_value, long_value = get_id(currency, soup)
			return_dict[currency] = [short_value, long_value]
		except AttributeError as e:
			print currency + ' not found???'
			continue

	return return_dict


def get_id(currency, html):
	symbol_dict = {'US':'USD','EU':'EUR', 'UK':'GBP', 'AL':'AUD', 'NZ':'NZD', 'JP':'JPY', 'MX':'MXN', 'SI':'SGD', 'SF':'ZAR', 'HK':'HKD', 'CA': 'CAD'}
	# Parse out the correct country 2 letter code
	country_1 = currency[3:5]
	country_2 = currency[5:7]
	# Create the new name using the symbol dictionary
	new_name = symbol_dict[country_2] + symbol_dict[country_1]
	short_id = 'rollshort' + new_name
	long_id = 'rolllong' + new_name
	# Search for the id 
	short_value = html.find(id=short_id)
	# If it does not exist, but be the modified version with 'A' appended
	if short_value is None:
		short_value = html.find(id=('A' + short_id))

	long_value = html.find(id=long_id)
	if long_value is None:
		long_value = html.find(id=('A' + long_id))

	return (short_value.text, long_value.text)
