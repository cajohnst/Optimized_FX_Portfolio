import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import datetime
from datetime import date
import os
import Pull_Data
import Daily_Reports 


on_heroku = False

if 'DYNO' in os.environ:
	on_heroku = True

def main():
	currency_list = Pull_Data.get_currency_list()
	wks = setup_credentials()

	if on_heroku:
		update_spreadsheet(wks, currency_list)
	else:
		request = raw_input('Enter Y to update the spreadsheet: ')
		if request is 'Y' or request is 'y':
			update_spreadsheet(wks, currency_list)


def setup_credentials():
	scope = ['https://spreadsheets.google.com/feeds']
	if on_heroku:
		keyfile_dict = setup_keyfile_dict()
		credentials = ServiceAccountCredentials.from_json_keyfile_dict(keyfile_dict, scope)
	else:
		credentials = ServiceAccountCredentials.from_json_keyfile_name('My Project-3b0bc29d35d3.json', scope)

	gc = gspread.authorize(credentials)

	if on_heroku:
		wks = gc.open_by_key("1DdmBaOlGGdgQRaaI3tQCxj3BEd8kPwaGIHVfMpIoH8I").sheet1
	else:

		wks = gc.open_by_key("1DdmBaOlGGdgQRaaI3tQCxj3BEd8kPwaGIHVfMpIoH8I").sheet1
	return wks

def setup_keyfile_dict():
	keyfile_dict = dict()
	keyfile_dict['type'] = os.environ.get('TYPE')
	keyfile_dict['client_email'] = os.environ.get('CLIENT_EMAIL')
	keyfile_dict['private_key'] = unicode(os.environ.get('PRIVATE_KEY').decode('string_escape'))
	keyfile_dict['private_key_id'] = os.environ.get('PRIVATE_KEY_ID')
	keyfile_dict['client_id'] = os.environ.get('CLIENT_ID')

	return keyfile_dict

def update_spreadsheet(wks, currency_list, weights_table):
	today = date.today()
	# If new spreadsheet, update current row indicator
	if wks.acell('A1').value == '':
		wks.update_acell('A1', 2)
	current_row = wks.acell('A1').value

	# calculate the last column based on the size of the weights table
	last_column = (increment_letter('B', len(weights_table)))

	if wks.acell('B1').value == '':
		populate_columns(wks, weights_table, last_column, currency_list)

	wks.update_acell('A' + current_row, today)

	cell_range = 'B' + current_row + ':' + last_column + current_row
	cell_list = wks.range(cell_range)
	for index, currency_data in enumerate(weights_table):
		cell_list[(index)].value = weights_table[index]
	wks.update_cells(cell_list)

	wks.update_acell('A1', int(current_row) + 1)

def populate_columns(wks, weights_table, last_column, currency_list):
	cell_list = wks.range('B1:' + last_column + '1') 
	for currency in currency_list:
		cell_list.value = currency 
	wks.update_cells(cell_list)

def pull_data(num_days):
    end_date = date.today()
    start_date = end_date - timedelta(num_days)
    wks = setup_credentials()
    
    csv_file = wks.export(format='csv')
    csv_buffer = StringIO.StringIO(csv_file)
    fxstreet_data = pd.read_csv(csv_buffer, header=1, index_col=0, parse_dates=True, infer_datetime_format=True)

    filtered_data = fxstreet_data.ix[start_date:end_date]

    return filtered_data


def increment_letter(letter, amount):
	cur = ord(letter)
	return chr(cur+amount)

if __name__ == "__main__":
	main()