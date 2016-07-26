import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
from datetime import date

from rollover_scraper import generate_rollover


def main():
	column_dictionary = {'USD/MXN': 'B', 'USD/CAD': 'D', 'NZD/USD': 'F', 'USD/HKD': 'H', 'USD/JPY': 'J', 'USD/SGD': 'L', 'GBP/USD': 'N', 'USD/ZAR': 'P', 'AUD/USD': 'R', 'EUR/USD': 'T'}


	scope = ['https://spreadsheets.google.com/feeds']
	credentials = ServiceAccountCredentials.from_json_keyfile_name('My Project-3b0bc29d35d3.json', scope)

	gc = gspread.authorize(credentials)

	wks = gc.open_by_key("1MW_NhhkPARpwtZfiLrn8v1EtzjQHLF5ifqkkWShFBO0").sheet1

	update_spreadsheet(wks, column_dictionary)

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



