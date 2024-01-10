"""
Web Scraping and QA Pair Generation Script

This script is designed to scrape web content, process it, and use OpenAI's GPT model to generate 
question-answer pairs based on the scraped content. It features a robust architecture with a WebScraper 
class for handling web requests, a DataProcessor class for processing links from a CSV file, and an 
OpenAIInterface class for interacting with the OpenAI API.

Key Components:
- WebScraper: Fetches and cleans webpage content.
- DataProcessor: Processes a list of URLs and uses the WebScraper to retrieve content.
- OpenAIInterface: Generates question-answer pairs based on the content using the OpenAI API.

Usage:
- Ensure necessary libraries are installed and .env file with API keys is correctly set up.
- Set the 'input_file_path' and 'output_file_path' in the configuration dictionary.
- Run the script to scrape content from webpages and generate QA pairs, which are then saved to a specified file.

Note:
- The script requires an OpenAI API key set in an .env file for generating QA pairs.
- The script is designed to handle errors and exceptions during web scraping and API interactions.
"""

import openai
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv

class WebScraper:
    def __init__(self):
        self.cache = {}

    def fetch_and_clean_webpage(self, url):
        if url in self.cache:
            return self.cache[url]

        try:
            response = requests.get(url)
            if response.status_code != 200:
                return "Failed to retrieve the webpage"
            soup = BeautifulSoup(response.text, 'html.parser')
            main_content = soup.find('main') or soup.body
            for script in main_content(["script", "style"]):
                script.decompose()
            cleaned_text = main_content.get_text(separator='\n', strip=True)
            self.cache[url] = cleaned_text
            return cleaned_text
        except Exception as e:
            return f"Error fetching page: {e}"

class DataProcessor:
    def __init__(self, scraper, cache_filename='06_Data/Capstone_Data/documentation_qa_datasets/content_cache.json'):
        self.scraper = scraper
        self.cache_filename = cache_filename

    def process_links(self, csv_file, limit=None):
        links_df = pd.read_csv(csv_file)
        text_links = links_df[links_df['Type'] == 'text-based']['LINK']

        self._load_cache()
        results = {}
        for i, link in enumerate(text_links):
            if limit and i >= limit:
                break
            content = self.scraper.fetch_and_clean_webpage(link)
            results[link] = content
            print(f"Processed {i+1}/{len(text_links)}: {link}")

        self._save_cache()
        return results

    def _load_cache(self):
        if os.path.exists(self.cache_filename):
            with open(self.cache_filename, 'r') as file:
                self.scraper.cache = json.load(file)

    def _save_cache(self):
        with open(self.cache_filename, 'w') as file:
            json.dump(self.scraper.cache, file)

class OpenAIInterface:
    def __init__(self, api_key):
        openai.api_key = api_key

    @staticmethod
    def create_qa_prompt(content):
        instructions = f"""
        Based on ONLY the contents below, please generate as many HIGH-QUALITY question answer pairs as there is information for. I want ONLY one of two responses below. Please make the question ONE SENTENCE and the answer ONE PARAGRAPH. I want you to focus on the MAIN IDEA of the articles for the questions. 

        FIRST CASE: If you determine that there IS enough information to produce a HIGH-QUALITY question answer pair, please return the answer in EXACTLY the format here:

        QUESTION: ...

        ANSWER: ...

        SECOND CASE: If you determine that there IS NOT enough information to produce a HIGH-QUALITY question answer pair, please return 'NOT ENOUGH INFORMATION'

        Here is the content of the webpage: {content}
        """
        return instructions

    def get_qa_response(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "system", "content": "Do EXACTLY as the instructions in the prompt say."},
                          {"role": "user", "content": prompt}]
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error in getting response: {e}")
            return None

# Main Execution
if __name__ == "__main__":
    # Configuration Dictionary
    config = {
        "input_file_path": "06_Data/Capstone_Data/documentation_qa_datasets/Test_Set_Links.csv", # Classified_VPC_Links <- Original pipeline dataset, replaced temp. to create the test dataset
        "output_file_path": "06_Data/Capstone_Data/documentation_qa_datasets/Test_Set_QA_Pairs.txt", # Documentation_QA_Pairs <- Original pipeline dataset, replaced temp. to create the test dataset
        "max_tokens": 1000,
        "test_mode": False,  # Set to False for full run
        "test_output_file": "06_Data/Capstone_Data/documentation_qa_datasets/Test_Set_QA_Pairs_Test.txt", # Documentation_QA_Pairs_Test <- Original pipeline dataset, replaced temp. to create the test dataset
        "test_limit": 5  # Number of links to process in test mode
    }

    # Load API key and create instances
    load_dotenv()
    openai_key = os.getenv("OPENAI_KEY")
    scraper = WebScraper()
    processor = DataProcessor(scraper)
    openai_interface = OpenAIInterface(openai_key)

    # Process links and get QA pairs
    limit = config['test_limit'] if config['test_mode'] else None
    processed_contents = processor.process_links(config['input_file_path'], limit=limit)
    output_file = config['test_output_file'] if config['test_mode'] else config['output_file_path']

    for url, content in processed_contents.items():
        prompt = OpenAIInterface.create_qa_prompt(content)
        qa_response = openai_interface.get_qa_response(prompt)

        with open(output_file, "a") as file:
            file.write(f"URL: {url}\nQ&A:\n{qa_response}\n\n")
        print(f"Processed URL: {url}")

    print(f"Output saved to {output_file}")



