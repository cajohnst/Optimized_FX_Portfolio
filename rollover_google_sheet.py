import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
from datetime import date
import os
import subprocess
from rollover_scraper import generate_rollover


def main():
	column_dictionary = {'USD/MXN': 'B', 'USD/CAD': 'D', 'NZD/USD': 'F', 'USD/HKD': 'H', 'USD/JPY': 'J', 'USD/SGD': 'L', 'GBP/USD': 'N', 'USD/ZAR': 'P', 'AUD/USD': 'R', 'EUR/USD': 'T'}

	scope = ['https://spreadsheets.google.com/feeds']
	keyfile_dict = setup_keyfile_dict()
	credentials = ServiceAccountCredentials.from_json_keyfile_dict(keyfile_dict, scope)

	gc = gspread.authorize(credentials)

	# wks = gc.open_by_key("1MW_NhhkPARpwtZfiLrn8v1EtzjQHLF5ifqkkWShFBO0").sheet1
	wks = gc.open_by_key("1IqTMl-yCH-X8GtkeuCVpV1UobDRc9V7ycxR19ySh5qI").sheet1

	subprocess.call(['./buildpack-run.sh'])

	update_spreadsheet(wks, column_dictionary)

def setup_keyfile_dict():

	keyfile_dict = dict()
	keyfile_dict['type'] = os.environ.get('TYPE')
	keyfile_dict['client_email'] = os.environ.get('CLIENT_EMAIL')
	keyfile_dict['private_key'] = unicode(os.environ.get('PRIVATE_KEY').decode('string_escape'))
	keyfile_dict['private_key_id'] = os.environ.get('PRIVATE_KEY_ID')
	keyfile_dict['client_id'] = os.environ.get('CLIENT_ID')


	return keyfile_dict

def update_spreadsheet(wks, column_dictionary):
	today= date.today()
	if wks.acell('A1').value == '':
		wks.update_acell('A1', 2)
	current_row= wks.acell('A1').value
	rollover_table= generate_rollover()

	if wks.acell('B1').value == '':
		populate_columns(wks, rollover_table, column_dictionary)
	wks.update_acell('A' + current_row, today)
	for name, short_val, long_val in rollover_table:
		current_column= column_dictionary[name]
		wks.update_acell(current_column + current_row, short_val)
		current_column= increment_letter(current_column)
		wks.update_acell(current_column + current_row, long_val)
	wks.update_acell('A1', int(current_row) + 1)


def populate_columns(wks, rollover_table, column_dictionary):
	for name, short_val, long_val in rollover_table:
		short_name = name + ' - S'
		long_name = name + ' - L'
		current_column = column_dictionary[name]
		wks.update_acell(current_column+'1', short_name)
		current_column = increment_letter(current_column)
		wks.update_acell(current_column+'1', long_name)


def increment_letter(letter):
	cur = ord(letter)
	return chr(cur+1)

if __name__ == "__main__":
	main()



