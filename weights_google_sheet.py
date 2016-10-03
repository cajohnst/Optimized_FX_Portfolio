import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
from datetime import date
import os
import Optimize_FX_Portfolio


on_heroku = False

if 'DYNO' in os.environ:
	on_heroku = True

def main():
	if date.today().weekday() == 1:
		currency_list = Optimize_FX_Portfolio.get_currency_list()
		currency_list.append("RF")

		weights_wks, merge_wks = setup_credentials()

		if on_heroku:
			update_spreadsheet(weights_wks, merge_wks)
		else:
			request = raw_input('Enter Y to update the spreadsheet: ')
			if request is 'Y' or request is 'y':
				update_spreadsheet(weights_wks, merge_wks)

def setup_credentials():
	scope = ['https://spreadsheets.google.com/feeds']
	if on_heroku:
		keyfile_dict = setup_keyfile_dict()
		credentials = ServiceAccountCredentials.from_json_keyfile_dict(keyfile_dict, scope)
	else:
		credentials = ServiceAccountCredentials.from_json_keyfile_name('My Project-3b0bc29d35d3.json', scope)

	gc = gspread.authorize(credentials)

	# Weights Table
	weights_wks = gc.open_by_key("1fibwcsUJOj9gWV6imgADLQuTuVeie_0ccNXIz01ZtuY").sheet1

	merge_wks = gc.open_by_key("10M7EDaurp43bBrCMQsdBVvQdjmXvhx_USvR4aWOAJ8M").sheet1
	return weights_wks, merge_wks

def setup_keyfile_dict():
	keyfile_dict = dict()
	keyfile_dict['type'] = os.environ.get('TYPE')
	keyfile_dict['client_email'] = os.environ.get('CLIENT_EMAIL')
	keyfile_dict['private_key'] = unicode(os.environ.get('PRIVATE_KEY').decode('string_escape'))
	keyfile_dict['private_key_id'] = os.environ.get('PRIVATE_KEY_ID')
	keyfile_dict['client_id'] = os.environ.get('CLIENT_ID')

	return keyfile_dict

def update_spreadsheet(weights_wks, merge_wks):
	today = date.today()

	weights_vector, merge_table = Optimize_FX_Portfolio.main()

	table_columns = list(merge_table.columns)

	# for day in range(4:9)[::-1]:
		



def update_weights(wks, weights_vector, table_columns):
	update_setup(wks)

	# calculate the last column based on the size of the rollover table
	last_column = increment_letter('B', len(weights_vector)-1)

	if wks.acell('B1').value == '':
		cell_list = wks.range('B1:' + last_column + '1')
		for index, cell in enumerate(cell_list):
			cell.value = table_columns[index]
		wks.update_cells(cell_list)
	wks.update_acell('A' + current_row, today)

	cell_range = 'B' + current_row + ':' + last_column + current_row
	cell_list = wks.range(cell_range)
	for index, cell in enumerate(cell_list):
		cell.value = weights_vector[index][0]
	wks.update_cells(cell_list)

	wks.update_acell('A1', int(current_row) + 1)


def update_merge(wks, merge_table, table_columns):
	update_setup(wks)

def update_setup(wks, last_column):
	today= date.today()
	if wks.acell('A1').value == '':
		wks.update_acell('A1', 2)
	current_row= wks.acell('A1').value

def pull_data():
	wks = setup_credentials()



def increment_letter(letter, amount):
	cur = ord(letter)
	return chr(cur+amount)

if __name__ == "__main__":
	main()



