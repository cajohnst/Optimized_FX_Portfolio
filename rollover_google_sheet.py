import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import datetime
from datetime import date
import os
from rollover_scraper import generate_rollover

on_heroku = False

if 'DYNO' in os.environ:
	on_heroku = True

def main():
	currency_list = get_currency_list()
	wks = setup_credentials()

	if on_heroku:
		update_spreadsheet(wks, currency_list)
	else:
		request = raw_input('Enter Y to update the spreadsheet: ')
		if request is 'Y' or request is 'y':
			update_spreadsheet(wks, currency_list)

def get_currency_list():
	currency_list = ['DEXMXUS', 'DEXCAUS', 'DEXUSNZ', 'DEXHKUS', 'DEXJPUS', 'DEXSIUS', 'DEXUSUK', 'DEXSFUS', 'DEXUSAL', 'DEXUSEU']
	return currency_list


def setup_credentials():
	scope = ['https://spreadsheets.google.com/feeds']
	if on_heroku:
		keyfile_dict = setup_keyfile_dict()
		credentials = ServiceAccountCredentials.from_json_keyfile_dict(keyfile_dict, scope)
	else:
		credentials = ServiceAccountCredentials.from_json_keyfile_name('My Project-3b0bc29d35d3.json', scope)

	gc = gspread.authorize(credentials)

	if on_heroku:
		wks = gc.open_by_key("1IqTMl-yCH-X8GtkeuCVpV1UobDRc9V7ycxR19ySh5qI").sheet1
	else:
		# wks = gc.open_by_key("1IqTMl-yCH-X8GtkeuCVpV1UobDRc9V7ycxR19ySh5qI").sheet1

		wks = gc.open_by_key("1MW_NhhkPARpwtZfiLrn8v1EtzjQHLF5ifqkkWShFBO0").sheet1
	return wks

def setup_keyfile_dict():
	keyfile_dict = dict()
	keyfile_dict['type'] = os.environ.get('TYPE')
	keyfile_dict['client_email'] = os.environ.get('CLIENT_EMAIL')
	keyfile_dict['private_key'] = unicode(os.environ.get('PRIVATE_KEY').decode('string_escape'))
	keyfile_dict['private_key_id'] = os.environ.get('PRIVATE_KEY_ID')
	keyfile_dict['client_id'] = os.environ.get('CLIENT_ID')

	return keyfile_dict

def update_spreadsheet(wks, currency_list):
	today = date.today()
	# If new spreadsheet, update current row indicator
	if wks.acell('A1').value == '':
		wks.update_acell('A1', 2)
	current_row = wks.acell('A1').value
	# generate the rollover table
	rollover_table = generate_rollover(currency_list)

	# calculate the last column based on the size of the rollover table
	last_column = (increment_letter('B', (len(rollover_table) * 2) - 1))

	if wks.acell('B1').value == '':
		populate_columns(wks, rollover_table, last_column)
	wks.update_acell('A' + current_row, today)

	cell_range = 'B' + current_row + ':' + last_column + current_row
	cell_list = wks.range(cell_range)
	for index, currency_data in enumerate(rollover_table):
		cell_list[(index * 2)].value = currency_data[1]
		cell_list[(index * 2) + 1].value = currency_data[2]
	wks.update_cells(cell_list)

	wks.update_acell('A1', int(current_row) + 1)

def populate_columns(wks, rollover_table, last_column):
	cell_list = wks.range('B1:' + last_column + '1')
	for index, currency_data in enumerate(rollover_table):
		short_name = currency_data[0] + ' - S'
		long_name = currency_data[0] + ' - L'
		cell_list[(index * 2)].value = short_name
		cell_list[(index * 2) + 1].value = long_name
	wks.update_cells(cell_list)

def pull_data(num_days):
	wks = setup_credentials()
	current_row = wks.acell('A1').value
	latest_entry = int(wks.acell('A1').value) - 1
	if num_days > latest_entry:
		start_row = 2
	else:
		start_row = latest_entry - num_days

	columns = [x for x in wks.row_values(1) if x]
	rollover_table = pd.DataFrame(columns=columns)
	for row_index in range(start_row, latest_entry + 1):
		row_data = pd.to_datetime(wks.row_values(row_index)[0])
		row_date = [float(x) for x in wks.row_values(row_index)[1:] if x]
		rollover_values = [row_data] + row_date
		rollover_table.loc[row_index - 2] = rollover_values
	rollover_table = rollover_table.set_index(current_row)
	return rollover_table


def increment_letter(letter, amount):
	cur = ord(letter)
	return chr(cur+amount)

if __name__ == "__main__":
	main()



