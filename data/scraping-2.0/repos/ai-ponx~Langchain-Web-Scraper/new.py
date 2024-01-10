import asyncio
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import openai
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
import os
import csv
from dotenv import load_dotenv

load_dotenv()

# Define the URL to scrape
url = ""

# Define the schema for LLM extraction
schema = {
    "properties": {
        "business_name": {"type": "string"},
        "phone_number": {"type": "string"},
        "address": {"type": "string", "format": "uri"},
        "email": {"type": "string", "format": "email"},  
    },
    "required": ["business_name", "phone_number", "address", "email"],
}

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def scrape_with_selenium(url):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)
    asyncio.sleep(5)  # Adjust as necessary
    html_content = driver.page_source
    driver.quit()
    return html_content

async def extract_and_process_data(url, schema):
    html_content = scrape_with_selenium(url)
    soup = BeautifulSoup(html_content, 'html.parser')

    # Targeted extraction based on provided classes
    targeted_elements = soup.select(".sc-gEvEer, .sc-jEACwC, .Anchor-sc-n6dour-1")

    # Extract text from targeted elements
    text_content = ' '.join([element.get_text() for element in targeted_elements])

    # Use LLM for advanced extraction
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", api_key=openai.api_key)
    extracted_content = await create_extraction_chain(schema=schema, llm=llm).run(text_content)

    return extracted_content

# Run the extractor and process the data with Langchain
processed_data = asyncio.run(extract_and_process_data(url, schema))

# Write the processed data to CSV
csv_file = 'extracted_data.csv'
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['business_name', 'phone_number', 'address', 'email'])
    writer.writeheader()
    for item in processed_data:
        writer.writerow(item)

print(f"Data written to {csv_file}")
