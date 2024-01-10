import gspread
from oauth2client.service_account import ServiceAccountCredentials as Credentials
import os
import openai
import time

# Set up Google Sheets API credentials
scope = ['https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_json_keyfile_name('client_s.json', scope)
# Authorize and open the Google Sheet

client = gspread.authorize(creds)
# Replace with your Google Sheet ID
sheet = client.open('#USE YOUR SHEET NAME HERE')
# Get all the titles from the 'title' column
# Assuming the titles are in the first worksheet
worksheet = sheet.get_worksheet(0)

company_names = worksheet.col_values(5)
company_data = worksheet.col_values(15)  # get the entire column data

# If company_data is shorter than company_names, append empty strings
while len(company_data) < len(company_names):
    company_data.append('')

# api for openai
openai.api_key = '#USE YOUR API KEY HERE'

# List to hold cells to be updated
cells_to_update = []

for i in range(len(company_names)):
    time.sleep(1)  # introduce delay
    # Get the current cell value
    cell_value = company_data[i]  # get value from local list
    # Check if the cell is empty
    if cell_value == '':
        prompt = "what do you know about the following company: " + \
            company_names[i] + "?"
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.6,
                max_tokens=100,
            )
        except openai.error.RateLimitError:
            print("Rate limit exceeded. Waiting for 60 seconds.")
            time.sleep(60)  # Wait for 60 seconds if rate limit is exceeded
            continue  # Skip to the next iteration

        # Add cell to the list to be updated
        cells_to_update.append(gspread.Cell(
            row=i+1, col=15, value=response.choices[0].text.strip()))
        # update the local list
        company_data[i] = response.choices[0].text.strip()

        # Update the Google Sheet in batches of 10 cells
        if len(cells_to_update) >= 10:
            worksheet.update_cells(cells_to_update)
            cells_to_update = []

# Update any remaining cells
if cells_to_update:
    worksheet.update_cells(cells_to_update)
