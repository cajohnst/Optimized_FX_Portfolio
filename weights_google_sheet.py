import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
from datetime import date
import os
import portfolio_optimization


on_heroku = False

if 'DYNO' in os.environ:
	on_heroku = True

def main():
	column_list = ['USD/MXN', 'USD/CAD', 'NZD/USD', 'USD/HKD','USD/JPY', 'USD/SGD', 'GBP/USD', 'USD/ZAR','AUD/USD','EUR/USD', 'RF']

	wks = setup_credentials()

	if on_heroku:
		update_spreadsheet(wks, column_list)
	else:
		request = raw_input('Enter Y to update the spreadsheet: ')
		if request is 'Y' or request is 'y':
			update_spreadsheet(wks, column_list)

def setup_credentials():
	scope = ['https://spreadsheets.google.com/feeds']
	if on_heroku:
		keyfile_dict = setup_keyfile_dict()
		credentials = ServiceAccountCredentials.from_json_keyfile_dict(keyfile_dict, scope)
	else:
		credentials = ServiceAccountCredentials.from_json_keyfile_name('My Project-3b0bc29d35d3.json', scope)

	gc = gspread.authorize(credentials)

	wks = gc.open_by_key("1fibwcsUJOj9gWV6imgADLQuTuVeie_0ccNXIz01ZtuY").sheet1

	return wks

def setup_keyfile_dict():
	keyfile_dict = dict()
	keyfile_dict['type'] = os.environ.get('TYPE')
	keyfile_dict['client_email'] = os.environ.get('CLIENT_EMAIL')
	keyfile_dict['private_key'] = unicode(os.environ.get('PRIVATE_KEY').decode('string_escape'))
	keyfile_dict['private_key_id'] = os.environ.get('PRIVATE_KEY_ID')
	keyfile_dict['client_id'] = os.environ.get('CLIENT_ID')

	return keyfile_dict

def update_spreadsheet(wks, column_list):
	today= date.today()
	if wks.acell('A1').value == '':
		wks.update_acell('A1', 2)
	current_row= wks.acell('A1').value

	weights_vector = portfolio_optimization.main()
	print weights_vector

	if wks.acell('B1').value == '':
		cell_list = wks.range('B1:L1')
		for index, cell in enumerate(cell_list):
			cell.value = column_list[index]
		wks.update_cells(cell_list)

	cell_range = 'B'+current_row+':L'+current_row
	cell_list = wks.range(cell_range)
	for index, cell in enumerate(cell_list):
		cell.value = weights_vector[index][0]
	wks.update_cells(cell_list)
	wks.update_acell('A' + current_row, today)
	# for name, short_val, long_val in rollover_table:
	# 	current_column= column_dictionary[name]
	# 	wks.update_acell(current_column + current_row, short_val)
	# 	current_column= increment_letter(current_column)
	# 	wks.update_acell(current_column + current_row, long_val)
	wks.update_acell('A1', int(current_row) + 1)

# def populate_columns(wks, rollover_table, column_dictionary):
# 	for name, short_val, long_val in rollover_table:
# 		short_name = name + ' - S'
# 		long_name = name + ' - L'
# 		current_column = column_dictionary[name]
# 		wks.update_acell(current_column+'1', short_name)
# 		current_column = increment_letter(current_column)
# 		wks.update_acell(current_column+'1', long_name)

def pull_data():
	wks = setup_credentials()



def increment_letter(letter):
	cur = ord(letter)
	return chr(cur+1)

if __name__ == "__main__":
	main()



