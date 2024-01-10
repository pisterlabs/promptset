import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import time
import re
import openai
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
global messages


def search_web(query, api_key):
    print(f"Search Query:\ {query}")

    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": 10}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return []

    search_results = response.json().get('webPages', {}).get('value', [])
    return [{'title': result['name'], 'url': result['url'], 'snippet': result['snippet']} for result in search_results]

def scrape_content(url):
    print("Scraping W B4")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return ""

    soup = BeautifulSoup(response.content, 'html.parser')
    print("Finished Scraping w/ b4")
    return soup.get_text()



import logging

def check_javascript_dependency(url):
    logging.basicConfig(level=logging.INFO)
    logging.info("Checking JavaScript dependency for URL: " + url)

    # Fetch content with requests
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text_requests = soup.get_text()

    # Set up the Selenium WebDriver
    service = Service(ChromeDriverManager().install())
    try:
        driver = webdriver.Chrome(service=service)
        driver.get(url)
        # Adjust the sleep time as necessary
        time.sleep(5)

        # Fetch content with Selenium
        text_selenium = driver.find_element('body').text

        # Compare the content length or look for specific markers
        if len(text_selenium) > len(text_requests) * 1.5:  # Adjust this threshold as needed
            logging.info("JavaScript is likely required for this URL.")
            return True

        if "enable JavaScript" in text_requests:
            logging.info("Found 'enable JavaScript' message in page source.")
            return True

    except WebDriverException as e:
        logging.error("WebDriver error: " + str(e))
        return False
    finally:
        driver.quit()

    logging.info("JavaScript does not seem to be required for this URL.")
    return False
def check_javascript_dependency_bak(url):
    # Set up the Selenium WebDriver
    service = Service(ChromeDriverManager().install())
    try:
        print("Currently checking for javascript dependency")
        driver = webdriver.Chrome(service=service)
        driver.get(url)

        # Fetch content with requests
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text_requests = soup.get_text()

        # Fetch content with Selenium
        text_selenium = driver.find_element('body').text

        # Compare the content length or look for specific markers
        if len(text_selenium) > len(text_requests) * 1.5:  # Arbitrary threshold
            driver.quit()
            return True  # Indicates reliance on JavaScript

        # Look for specific messages in HTML
        if "enable JavaScript" in text_requests:
            driver.quit()
            return True

    except WebDriverException as e:
        print(f"WebDriver error: {e}")
        driver.quit()
        return False
    finally:
        driver.quit()

    return False

def scrape_content_javascript_included(url):
    if check_javascript_dependency(url):
        print("Javascript Found: Scraping w/ selenium")
    # Use Selenium
        response = scrape_content_with_selenium(url)
        return response
    else:
        print("Jacascript Not Found: Scraping w/ requests")
        response = scrape_content_b4(url)
        return response


# Use requests and BeautifulSoup



def scrape_content_with_selenium(url):
    # Set up the Selenium WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    try:
        print("Trying to scrape w/ selenium")
        driver.get(url)
        time.sleep(5)  # Wait for JavaScript to load

        # Get page source and close the browser
        page_source = driver.page_source
        driver.quit()

        # Use BeautifulSoup to parse the page source
        soup = BeautifulSoup(page_source, 'html.parser')
        return soup.get_text()

    except Exception as e:
        print(f"Error scraping with Selenium: {e}")
        driver.quit()
        return ""

def create_search_file(search_results):
    with open('search_results_temp.txt', 'w', encoding='utf-8') as file:
        for result in search_results:
            file.write(f"Title: {result['title']}\n")
            file.write(f"URL: {result['url']}\n")
            file.write("Content:\n")
            content = scrape_content(result['url'])

            # Split content into lines, filter out empty lines, and rejoin
            filtered_content = '\n'.join([line for line in content.splitlines() if line.strip()])

            file.write(filtered_content)
            file.write("\n" + "#" * 50 + "\n")



#######################################################TEST


def process_query_and_respond(file_path, query):
    client = OpenAI()
    delete_all_temp_files(client)
    delete_all_assistants(client)
    time.sleep(2)
    file_path = 'search_results_temp.txt'
    file = client.files.create(
        file=open(file_path, "rb"),
        purpose='assistants'
    )
    file_id = file.id
    print(f"File Info:\n {file}\n")

    # Create an assistant with retrieval enabled

    # Add the file to the assistant

    assistant = client.beta.assistants.create(
        name="Bernard Search",
        instructions="You are a web search results analyzer who answers questions based on websearch results stored "
                     "in a text file. These search results are generated using webscraping tools so be sure to check"
                     "thoroughly before answering, you may have to wade through some clutter to infer the correct"
                     "response based on search results.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids=[file_id]
    )
    print(f"Assistant Info:\n {assistant}\n")
    assistant_id = assistant.id

    assistant = client.beta.assistants.retrieve(f"{assistant_id}")
    print("Assistant Located")

    thread = client.beta.threads.create()
    print("Thread  Created")
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"I need you respond to the following user input using the file uploaded earlier{query}"
    )
    print("Thread Ready")

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    print("Assistant Loaded")
    print("Run Started - Please Wait")

    while True:
        time.sleep(10)

        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

        if run_status.status == "completed":
            print("Run is Completed")
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            if messages.data:
                content = messages.data[0].content[0].text.value
                cleaned_content = re.sub(r"【\d+†source】", "", content)
                cleaned_content = cleaned_content.strip()
                #print(cleaned_content)
                #delete_all_temp_files(client)
                return cleaned_content
            else:
                print("No messages received.")
            # break
            time.sleep(1)
            #delete_all_temp_files(client)
            # print(messages.data. content[0].text.value)
            break
        else:
            print("Run is in progress - Please Wait")
            continue
def delete_all_assistants(client):
    # Retrieve all assistants
    assistants = client.beta.assistants.list(limit=None)

    # Loop through each assistant and delete
    for assistant in assistants.data:
        try:
            response = client.beta.assistants.delete(assistant.id)
            print(f"Deleted assistant {assistant.id}: {response}")
        except openai.error.OpenAIError as e:
            print(f"Failed to delete assistant {assistant.id}: {e}")
def delete_all_temp_files(client):
    # List all files
    files = client.files.list()

    # Iterate through the files and delete 'search_results_temp.txt'
    for file in files:
        if file.filename == 'search_results_temp.txt':
            try:
                # Delete the file
                client.files.delete(file.id)
                print(f"Deleted file: {file.id} - {file.filename}")
            except Exception as e:
                print(f"Error deleting file {file.id}: {e}")

def main(search_query):
    # Initialize OpenAI API key
    load_dotenv("config.env")
    openai.api_key = os.environ.get("OpenAiKey")
    # Example Usage
    api_key = os.environ.get("bing_api")

    search_results = search_web(search_query, api_key)
    print(f"Search Results:\n {search_results}\n")
    if search_results:
        create_search_file(search_results)
        search_results_path = 'search_results_temp.txt'
        response = process_query_and_respond(search_results_path, search_query)
        #print(f"Response: \n{response}\n")
        return response

    else:
        error_text = "no search results found :("
        print(error_text)
        return error_text


if __name__ == "__main__":
    search_query = "what movies are playing in houghton michigan today? 1/4/2024"  # input("Enter your search query: ")
    search_results_path = 'search_results_temp.txt'
    response = main(search_query)
    print(f"The Response\n {response}")

    

