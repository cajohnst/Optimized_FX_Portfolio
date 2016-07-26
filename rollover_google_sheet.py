import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds']
credentials = ServiceAccountCredentials.from_json_keyfile_name('My Project-3b0bc29d35d3.json', scope)

gc = gspread.authorize(credentials)

wks = gc.open_by_key("1MW_NhhkPARpwtZfiLrn8v1EtzjQHLF5ifqkkWShFBO0").sheet1

wks.update_acell('A1', 'testing')