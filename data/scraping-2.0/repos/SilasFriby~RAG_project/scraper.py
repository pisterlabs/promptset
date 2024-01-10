import pprint
import os
import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import AsyncChromiumLoader
from playwright.sync_api import sync_playwright
from langchain_core.documents.base import Document
from typing import Sequence

def handle_cookies(page):
    # Wait for the cookie consent "Allow all" button to appear and click it
    # The selector used here is based on the provided screenshot
    allow_all_cookies_selector = '#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll'
    page.wait_for_selector(allow_all_cookies_selector)
    page.click(allow_all_cookies_selector)

def login(page, username, password):
    # Navigate to the main page
    page.goto('https://www.8a.nu/')

    # Handle cookies if the consent popup appears
    handle_cookies(page)

    # Click the login link to open the login form
    page.click('text="Log in"')

    # Wait for the username field to be visible
    page.wait_for_selector('input#username')

    # Fill in the username and password fields
    page.fill('input#username', username)
    page.fill('input[name="password"]', password)

    # Click on the submit button for the login form
    page.click('input#kc-login')  # Make sure this is the correct selector for the login button

    # Wait for one second to allow the page to load
    page.wait_for_timeout(1000)

    # # Wait for navigation to confirm login
    # page.wait_for_url('https://www.8a.nu/')

def search_and_select_first_result(page, search_query):
    # Click the search input to focus it
    search_button_selector = '.search-btn'
    page.click(search_button_selector)

    # Type the search query into the search input
    page.fill('input[placeholder="Search for areas, crags, routes"]', search_query)

    # Wait for the search results to appear and click on the first result
    # The exact selector for the first search result will depend on the website's structure
    # Replace 'div[data-v-xxxxxx]:first-child' with the correct selector
    page.wait_for_selector('div.result-item-title')  # Example selector, replace with the actual one
    page.click('div.result-item-title')

    page.wait_for_timeout(1000)

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = openai_api_key


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

schema = {
    "properties": {
        "NAME_column": {"type": "string"},
        "ASCENTS_column": {"type": "integer"},

    },
    "required": ["NAME_column", "ASCENTS_column"],
}


def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)

def scrape_with_playwright(page, schema):
    content = page.content()

    # Create variable doc based on class Sequence from typing.py and class Document from base.py
    docs = [Document(page_content=content)]

    # loader = AsyncChromiumLoader(url)
    # docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["td"]
    )
    print("Extracting content with LLM")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)

    # Process the first split
    extracted_content = extract(schema=schema, content=splits[0].page_content)
    pprint.pprint(extracted_content)
    return extracted_content

# Main execution
def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set headless=True for no UI
        page = browser.new_page()
        username = 'silasfriby@gmail.com'   # Replace with your actual username
        password = 'friby1'   # Replace with your actual password
        search_query = "San Vito lo Capo"

        login(page, username, password)
        search_and_select_first_result(page, search_query)

        extracted_content = scrape_with_playwright(page, schema=schema)
    
        browser.close()

if __name__ == '__main__':
    main()



