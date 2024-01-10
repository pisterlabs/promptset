import csv
from itertools import count
from dotenv import load_dotenv
load_dotenv()

import time 


import requests
from bs4 import BeautifulSoup

import os
import openai

# Assuming your CSV file is named 'your_file.csv'
csv_file_path = 'Data.csv'

def add_data_to_model():
    # Open the CSV file with 'latin-1' encoding
    row_count = 0
    rows_to_process = 25

    with open(csv_file_path, 'r', encoding='latin-1') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        # Iterate over rows in the CSV file
        for row in csv_reader:
            # Unpack values into variables
            id, problem, solution = row 

            # Now you can use id, problem, solution as separate variables
            #print("ID:", id)
            #print("Problem:", problem)
            #print("Solution:", solution)
            idea = "problem: " + problem + ", solution: "+ solution 
            message_list.append( {"role": "system", "content": idea})
            # Increment the row count
            row_count += 1

            # Check if the desired limit is reached
            if row_count >= rows_to_process:
                break
api_key = os.getenv("API_KEY7")

def data_scrape(url, timeout=1):
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if response.status_code == 200:
            print("Successful response")
            if elapsed_time > timeout:
                #print("Request took more than 2 seconds. Skipping...")
                return -1
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.text
        else:
            #print('Failed to retrieve the page. Status code:', response.status_code)
            return -1
    except requests.exceptions.Timeout:
        #print('Request timed out after', timeout, 'seconds. Skipping...')
        return -1
    except Exception as e:
        #print('An error occurred:', str(e))
        return -1




def search_websites_with_keyword(keyword):
    # Define the search query
    search_query = f"intitle:{keyword}"  # This query searches for "rpi" in the title of web pages

    # Send a GET request to Google Search with the query
    search_url = f"https://www.google.com/search?q={search_query}"
    headers = {"User-Agent": "Your User Agent Here"}  # Replace with your User-Agent header
    response = requests.get(search_url, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the search results page
        soup = BeautifulSoup(response.text, "html.parser")

        # Find and extract search result links
        search_results = soup.find_all("a")
        counter = 0 
        while counter < 5:
            for result in search_results:
                link = result.get("href")
                if link and link.startswith("/url?q="):
                    # Extract the actual URL from the Google search result link
                    url = link[7:]  # Remove "/url?q=" prefix
                    info = data_scrape(url)
                    if (info != -1 ):
                        message_list.append( {"role": "system", "content": info})
                        print(len(message_list))
                        counter+=1


                



                # You can further process this URL or send a request to scrape data from the website

    else:
        print("Failed to retrieve search results.")



message_list=[
    {"role": "system", "content": "You are an idea validator. You advice human evaulators by devloping clear rationale and reatings for essential metrics such as maturity stage, market potential, feasibility, scalability, technological innovation, or adherence to circular ecnonomy principles. Ideas that meet(self-)predefined criteria will be highlighted to human evaluators. You basically have to highlight or emphasize solutions that pass the threshold. These will be given as user input messages. Show the rating on the necessary factors and say the idea along with the problem and solution"},
  ]


def create_response(user_input):
    message_list.append( {"role": "user", "content": user_input})

    openai.api_key = api_key
    completion = openai.ChatCompletion.create(
    model="gpt-4-0314",
    messages=message_list
   
    )
  
    print(completion.choices[0].message.content)


add_data_to_model()

user_input = input("Enter a message: ")
#search_websites_with_keyword("rpi")

#ex_url = "https://www.rpi.edu/"
#some_info = data_scrape(ex_url)
#message_list.append( {"role": "system", "content": some_info})
#message_list.pop() 
create_response(user_input)


