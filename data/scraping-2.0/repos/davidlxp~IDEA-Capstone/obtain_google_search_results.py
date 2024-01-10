# obtain_google_search_results.py
# [Developer] Xiaotian Cao
# [How to run] Drag this script to any IDE like Spyder, and run the current script.
#
# [Description] When running this python file, the program will open Goolge and search for
# "movie/tv show name" + "free watch online", collect domains, and return a list of websites
# in a decreasing order in frequency of appearance.
import time
import pandas as pd
import csv
from openai import OpenAI

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService

system_instruction = 'Translate the given English string into '
driver_path = "chromedriver.exe"
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument(
    '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')
client = OpenAI(api_key="openai_key")


def getGoogleResults(name):
    # Set the path to the WebDriver executable
    driver = webdriver.Chrome()

    # Open Google in the web browser
    driver.get('https://www.google.com')
    wait = WebDriverWait(driver, 10)

    # # Find the search input element and enter your query
    search_box = driver.find_element(By.ID, 'APjFqb')
    search_box.send_keys(name)

    # # Submit the search form
    search_box.submit()

    # Create a list of length 50 to store the href values
    hrefs = []
    max = 50
    count = 0

    # Scroll down and wait for new results to load (you may need to adjust the scrolling and waiting logic)
    for _ in range(10):  # Scroll 3 times, you can adjust the number
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        new_results = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a[jsname="UWckNb"]')))
        for result in new_results:
            href = result.get_attribute('href')
            if href and count < max:
                hrefs.append(href)
                count += 1

    # Close the browser
    driver.quit()

    return hrefs


def findIpAddress(url):
    # Find the starting index of "https://"
    start_index = url.find("https://")

    # Check if "https://" was found in the URL
    if start_index != -1:
        # Find the index of the next "/"
        end_index = url.find("/", start_index + len("https://"))

        # Check if a "/" was found after "https://"
        if end_index != -1:
            # Extract the substring between "https://" and the next "/"
            extracted_substring = url[start_index + 8:end_index]

            if extracted_substring.startswith('www'):
                extracted_substring = extracted_substring[4:]

            # Print the extracted substring
            return extracted_substring
        else:
            return
    else:
        return


def translateMovieName(name, language):
    # Create the content string for passing to GPT
    user_content = name + " watch free online"
    messages = [
        {"role": "system", "content": system_instruction + language},
        {"role": "user", "content": user_content}
    ]
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages
    )
    return chat.choices[0].message.content


def write_dict_to_csv(data_dict, csv_file_path):
    # Open the CSV file in write mode
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        # Extract keys and values separately
        keys = list(data_dict.keys())
        values = list(data_dict.values())

        # Create a CSV writer
        csv_writer = csv.writer(csv_file)

        # Write the header to the CSV file
        csv_writer.writerow(['domain', 'freq'])

        # Write the data to the CSV file
        csv_writer.writerows(zip(keys, values))


def main():
    website_counts = {}
    movies = pd.read_csv("IMDB-Movie-Data.csv")
    movie_names = list(movies["Title"][0: 10])

    for search_name in movie_names:
        for language in ['English', 'Chinese', 'Spanish', 'Indi']:
            search_string = translateMovieName(search_name, language)
            search_results = getGoogleResults(search_string)
            for result in search_results:
                url = findIpAddress(result)
                if url not in website_counts:
                    website_counts[url] = 0
                    # Increment the count for the element
                website_counts[url] += 1
    # Sort the dictionary by values (in ascending order)
    sorted_dict = dict(sorted(website_counts.items(), key=lambda item: item[1], reverse=True))

    # Print the sorted dictionary by values
    write_dict_to_csv(sorted_dict, "output.csv")


if __name__ == "__main__":
    main()
