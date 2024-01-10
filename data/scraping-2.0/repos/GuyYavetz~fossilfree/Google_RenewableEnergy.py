from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import time
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials as Credentials
import os
import openai

# Set up Google Sheets API credentials
scope = ['https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_json_keyfile_name('client_s.json', scope)
# Authorize and open the Google Sheet
client = gspread.authorize(creds)
# Replace with your Google Sheet ID
sheet = client.open('#USE YOUR SHEET NAME HERE')

worksheet = sheet.get_worksheet(0)

company_names = worksheet.col_values(5)

driver_path = r"#USE YOUR DRIVER PATH HERE"
driver = webdriver.Chrome(executable_path=driver_path)

# WebDriverWait instance
wait = WebDriverWait(driver, 10)  # wait for up to 10 seconds

try:
    for i in range(1, len(company_names), 10):
        # Initialize an empty list to store the results for this batch
        batch_results = []

        # Process the next 10 companies (or fewer if fewer than 10 remain)
        for company in company_names[i:i+10]:
            # Remove leading and trailing dots
            company = company.strip('.')

            driver.get('https://www.google.com')

            try:
                search_box = wait.until(EC.element_to_be_clickable((By.NAME, 'q')))
                search_query = company + ' Renewable energy'
                search_box.send_keys(search_query)
                search_box.send_keys(Keys.RETURN)
            except StaleElementReferenceException:
                continue

            time.sleep(2)  # wait for the page to load

            result_stats = driver.find_element(By.ID, 'result-stats').text
            num_results = re.search(r'(\d+(,\d{3})*)', result_stats)
            if num_results:
                print(f"Company: {company}, Results: {num_results.group(1)}")
                # Store the result for this company in the batch_results list
                batch_results.append(num_results.group(1))

        worksheet.update('Q' + str(i+2) + ':Q' + str(i+2+len(batch_results)-1), [[result] for result in batch_results])

finally:
    driver.quit()
