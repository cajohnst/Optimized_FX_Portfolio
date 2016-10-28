import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import datetime
from datetime import date, timedelta 
import StringIO
import csv
import os
import Pull_Data 
import settings as sv 

on_heroku = False

if 'DYNO' in os.environ:
    on_heroku = True

def main(weights=None, sheet_name=None):
    currency_list = Pull_Data.get_currency_list()
    # Append RF to list
    # currency_list.append('RF')
    sps = setup_credentials()

#     if on_heroku:
#         update_spreadsheet(wks, currency_list)
#     else:
#         request = raw_input('Enter Y to update the spreadsheet: ')
#         if request is 'Y' or request is 'y':
#             update_spreadsheet(wks, currency_list)
    wks = sps.worksheet(sheet_name)
    update_spreadsheet(wks, currency_list, weights)

def setup_credentials():
    scope = ['https://spreadsheets.google.com/feeds']
    if on_heroku:
        keyfile_dict = setup_keyfile_dict()
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(keyfile_dict, scope)
    else:
        credentials = ServiceAccountCredentials.from_json_keyfile_name('My Project-3b0bc29d35d3.json', scope)

    gc = gspread.authorize(credentials)

    if on_heroku:
        sps = gc.open_by_key("1DdmBaOlGGdgQRaaI3tQCxj3BEd8kPwaGIHVfMpIoH8I")
    else:
        sps = gc.open_by_key("1DdmBaOlGGdgQRaaI3tQCxj3BEd8kPwaGIHVfMpIoH8I")
    return sps

def setup_keyfile_dict():
    keyfile_dict = dict()
    keyfile_dict['type'] = os.environ.get('TYPE')
    keyfile_dict['client_email'] = os.environ.get('CLIENT_EMAIL')
    keyfile_dict['private_key'] = unicode(os.environ.get('PRIVATE_KEY').decode('string_escape'))
    keyfile_dict['private_key_id'] = os.environ.get('PRIVATE_KEY_ID')
    keyfile_dict['client_id'] = os.environ.get('CLIENT_ID')

    return keyfile_dict

def bootstrap_sheet(wks):
    # If new spreadsheet, update current row indicator
    if wks.acell('A1').value == '':
        wks.update_acell('A1', 2)

def update_spreadsheet(wks, currency_list, weights):
    bootstrap_sheet(wks)
    current_row = int(wks.acell('A1').value)

    # calculate the last column based on the size of the weights table
    last_column = (increment_letter('B', len(weights)))

    if wks.acell('B1').value == '':
        populate_columns(wks, last_column, currency_list)
    # Populate date
    wks.update_acell('A' + str(current_row), sv.end_date)

    cell_range = 'B' + str(current_row) + ':' + last_column + str(current_row)
    cell_list = wks.range(cell_range)
    for index, weight in enumerate(weights):
        cell_list[index].value = weight
    wks.update_cells(cell_list)

    wks.update_acell('A1', current_row + 1)

def populate_columns(wks, last_column, currency_list):
    cell_list = wks.range('B1:' + last_column + '1') 
    for index, currency in enumerate(currency_list):
        cell_list[index].value = currency 
    wks.update_cells(cell_list)

def pull_data(num_days, sheet_name):
    start_date = sv.end_date  - timedelta(num_days)
    sps = setup_credentials()
    wks = sps.worksheet(sheet_name)
    
    csv_file = wks.export(format='csv')
    csv_buffer = StringIO.StringIO(csv_file)
    weights_data = pd.read_csv(csv_buffer, header= 0, index_col=0, parse_dates=True, infer_datetime_format=True)

    earliest_date = weights_data.index[0].date()
    if start_date < earliest_date:
        start_date = earliest_date

    filtered_data = weights_data.ix[start_date:sv.end_date]

    return filtered_data

def increment_letter(letter, amount):
    cur = ord(letter)
    return chr(cur+amount)

if __name__ == "__main__":
    main()