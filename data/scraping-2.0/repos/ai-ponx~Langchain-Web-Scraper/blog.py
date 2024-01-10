import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import openai
from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
import csv
import os
from dotenv import load_dotenv

load_dotenv()

# Enter the URL
url = "ENTER URL HERE"

# Define the schema you wish to extract
schema = {
    "properties": {
        "business_name": {"type": "string"},
        "phone_number": {"type": "string"},
        "address": {"type": "string", "format": "uri"},  
    },
    "required": ["business_name", "phone_number", "address"],
}


# Set up OpenAI API 
openai.api_key = os.getenv("OPENAI_API_KEY")

async def scrape_with_playwright(url, schema):
    # Start Playwright in asynchronous mode
    async with async_playwright() as p:
        # Launch a headless Chromium browser
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)

        # Get the HTML content of the page
        html_content = await page.content()
        await browser.close()

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Define tags to extract
        tags_to_extract = ['h1', 'h2', 'h3', 'p', 'li', 'div', 'span', 'a']

        # Extract text from the defined tags
        text_content = ' '.join([element.get_text() for tag in tags_to_extract for element in soup.find_all(tag)])

        # Use LLM for advanced extraction
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
        extracted_content = create_extraction_chain(schema=schema, llm=llm).run(text_content)

        return extracted_content

# Run the scraper
extracted_content = asyncio.run(scrape_with_playwright(url, schema))

# Writing to CSV
csv_file = 'output.csv'
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['business_name', 'phone_number', 'address'])
    writer.writeheader()
    for item in extracted_content:
        writer.writerow(item)

print(f"Data written to {csv_file}")
