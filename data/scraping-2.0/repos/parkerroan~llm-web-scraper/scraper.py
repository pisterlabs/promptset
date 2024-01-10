import argparse
import pprint
import json
import csv 
from bs4 import BeautifulSoup, NavigableString
from playwright.sync_api import sync_playwright
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from dotenv import load_dotenv
from urllib.parse import urlparse  # <-- Added this import


def clean_html_extract_text(soup: BeautifulSoup) -> str:
    # Remove script and style elements
    for script in soup(['script', 'style']):
        script.extract()

    # Iterate over all img tags and replace them with their src and alt attributes
    for img in soup.find_all('img'):
        if img.has_attr('src'):
            alt_text = img.get('alt', '')  # Get alt text, default to an empty string if not present
            new_content = f"[Image: {alt_text}] {img['src']}"
            new_tag = NavigableString(new_content)
            img.replace_with(new_tag)

    # Iterate over all a tags and replace them with their href attribute and inner text
    for a in soup.find_all('a'):
        if a.has_attr('href'):
            link_text = a.get_text(strip=True)
            new_content = f"[Link: {link_text}] {a['href']}"
            new_tag = NavigableString(new_content)
            a.replace_with(new_tag)

    # Get the modified content
    modified_content = soup.get_text(strip=True)

    # Remove multiple line breaks and spaces
    return modified_content

def extract_elements_from_content(page_source: str, element_selector: str = None) -> list:
    soup = BeautifulSoup(page_source, "html.parser")

    story_elements = None
    if element_selector is None:
        story_elements = soup.body.select(element_selector)
    else: 
        story_elements = soup.body.select("article, .post") # default which may work for some news sites

    els = [clean_html_extract_text(BeautifulSoup(str(el), 'html.parser')) for el in story_elements]
    return els

def fetch_webpage_content(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        user_agent = ("Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Mobile Safari/537.36")
        page = browser.new_page(user_agent=user_agent)
        page.goto(url, timeout=100000)
        page_source = page.content()
        browser.close()
    return page_source

def load_schema_from_file(filename: str) -> dict:
    with open(filename, 'r') as file:
        schema = json.load(file)
    return schema

def main(url: str, element_selector: str = None):
    page_source = fetch_webpage_content(url)
    els = extract_elements_from_content(page_source, element_selector=element_selector)

    # Convert to string and separate by separator
    content = "\n<-->\n".join([str(el) for el in els if el])

    # Split the content into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=0,
        separators=["\n<-->\n"],
        length_function=len,
        add_start_index=True,
    )

    # Create documents
    texts = splitter.create_documents([content])

    # Schema
    schema = load_schema_from_file("schema.json")

    template = """
    Extract and save the relevant entities mentioned in the following passage together with their properties.

    Only extract the properties mentioned in the 'information_extraction' function.

    For context, the unstructured text in the passage has been pulled from html elements on a webpage.

    If a property is not present and is not required in the function parameters, do not include it in the output. 

    Passage:
    {input}
    """
    prompt=ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

    chain = create_extraction_chain(schema, llm, prompt=prompt, verbose=True)
    output = chain.run(texts[0])

    return output

if __name__ == "__main__":
    load_dotenv()  # take environment variables from .env.

    parser = argparse.ArgumentParser(description='Web scraper and story extractor.')
    parser.add_argument('url', type=str, help='The URL to scrape content from. ex: "https://www.npr.org"')
    parser.add_argument('--selector', type=str, default="[data-component-name='card']", help='The CSS selector to identify the elements to extract. ex: "[class~="story-wrap"]"')
    args = parser.parse_args()
    
    output = main(args.url, element_selector=args.selector)

    parsed_url = urlparse(args.url)
    # Take the netloc (domain) and replace non-alphanumeric characters with underscores
    filename = parsed_url.netloc.replace('.', '_').replace('-', '_')
    
    # Write to CSV
    with open(f"{filename}.csv", 'w', newline='') as csvfile:
        fieldnames = ['title', 'summary', 'image_url', 'link']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for content in output:
            json_formatted_str = json.dumps(content, indent=2)
            pprint.pprint(json_formatted_str)
            writer.writerow(content)

