from openai import OpenAI
import pandas as pd
import csv
import os
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import re

chrome_options = webdriver.ChromeOptions()
#chrome_options.add_argument('--headless')  # Add this line to run in headless mode

# Initialize Chrome WebDriver with the specified options
driver = webdriver.Chrome(options=chrome_options)

inputCSV = '/Users/MAC/Desktop/Ex-3-1.csv'
outputCSV = '/Users/MAC/Desktop/resultstest.csv'
df = pd.read_csv(inputCSV)

output_directory = os.path.dirname(outputCSV)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

with open(outputCSV, 'a', newline='', encoding='utf-8') as csv_file:
    fieldnames = ['yelp_link', 'address', 'city', 'province', 'country']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Write header only if the file is empty
    if csv_file.tell() == 0:
        csv_writer.writeheader()

existing_urls = set()
if os.path.exists(outputCSV):
    with open(outputCSV, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            existing_urls.add(row['yelp_link'])

column_a_data = []
print(df.columns)
for num in range(0,len(df)):
    row = df.iloc[num]
    column_a_data.append([{'name': row['yelp_link'],
                           'reviewsURL': row['reviewsURL']
    }])

column_a_data = [row for row in column_a_data if row[0]['name'] not in existing_urls if row[0]['name'] != '' and row[0]['name'] != 'yelp_link']

for row in column_a_data:  # Adjust 'k' as needed
    row = row[0]
    url, reviewURL = row['name'],row['reviewsURL']
    print(row['name'])
    try:
        match = re.search(r'@([0-9.-]+,[0-9.-]+,[0-9]+z)', reviewURL)
        coordinates = match.group(1)
        # Split the coordinates and take the first two values
        lat, lon = coordinates.split(',')[:2]

        # Format the coordinates with specific precision
        coordinates = f'{float(lat):.5f},{float(lon):.5f}'
    except:
        results = []
        result_entry = {
            'yelp_link': row['name'],
            'address': row['reviewsURL'],
            'country': '',
            'province': '',
            'city': ''
        }

        results.append(result_entry)

        print(result_entry)
        with open(outputCSV, 'a', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['yelp_link', 'address', 'city', 'province', 'country']
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writerow(result_entry)

        continue

    print(coordinates)
    results = []
    driver.get('https://www.bing.com/maps')
    # Enter a location in the search bar
    time.sleep(2)
    searchBox = driver.find_element(By.CSS_SELECTOR,
                                     'input[id="maps_sb"]')
    searchBox.send_keys(coordinates)

    time.sleep(2)
    searchButton = driver.find_element(By.CSS_SELECTOR, 'div[class="searchButton"]')
    searchButton.click()
    time.sleep(3)

    expandAddress = driver.find_element(By.CSS_SELECTOR, 'button[class="geochainActionButton geochainUncollapse"]')
    expandAddress.click()
    time.sleep(2)
    # Get the text content inside the div element
    try:
        country = driver.find_element(By.CSS_SELECTOR, 'a[class="geochainSegment"][data-index="0"]').text
    except:
        country = ""

    try:
        province = driver.find_element(By.CSS_SELECTOR, 'a[class="geochainSegment"][data-index="1"]').text
    except:
        province = ""

    try:
        city = driver.find_element(By.CSS_SELECTOR, 'a[class="geochainSegment"][data-index="2"]').text
    except:
        city = ""

    result_entry = {
        'yelp_link': row['name'],
        'address': row['reviewsURL'],
        'country': country,
        'province': province,
        'city': city
    }
    results.append(result_entry)

    print(result_entry)
    with open(outputCSV, 'a', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['yelp_link', 'address', 'city', 'province', 'country']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writerow(result_entry)
