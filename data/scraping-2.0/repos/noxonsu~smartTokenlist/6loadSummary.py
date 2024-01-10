import json
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
from langchain.schema import SystemMessage, HumanMessage
import os
from langchain.chat_models import ChatOpenAI
from bs4 import BeautifulSoup
import re
from utils import *
MAINFILE = os.environ.get("MAINFILE")
NETWORK = os.environ.get("NETWORK")

def load_data_from_file(filename):
    with open(filename, "r") as f:
        return json.load(f)

def filter_sites_without_summary(data):
    return [
        (entry['web_domains'][0], entry['contract_address'])
        for entry in data 
        if not entry.get('processedGpt') and 
        not entry.get('p6') and 
        entry.get('web_domains') and
        entry.get('holders') and 
        entry['holders'].get(NETWORK, float('inf')) < 500  # This line ensures the 'holders' count is less than 
    ]

def extract_content(site):
    loader = AsyncChromiumLoader(["https://"+site])
    docs = loader.load()
    telegram_links = extract_telegram_links(docs[0].page_content)
    email = exctract_email(docs[0].page_content)
    html2text = Html2TextTransformer()
    text_content = html2text.transform_documents(docs)
    if not text_content:
        return "Failed to extract content for site {site}", telegram_links
    return text_content, telegram_links, email

def extract_telegram_links(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    regex = re.compile(r"https?://t\.me/(\w+)")
    links = soup.find_all('a', href=regex)
    telegram_groups = [regex.search(link['href']).group(1) for link in links]
    return telegram_groups

def exctract_email(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Regex to match email addresses
    regex = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

    # Searching only within certain tags
    for tag in soup.find_all(['p', 'div', 'a']):
        email = tag.find(string=regex)
        if email:
            return email

    return None

def save_summary_and_proposal(contract_address, summary):
    if not os.path.exists('summaries'):
        os.makedirs('summaries')
    filename = os.path.join('summaries', f"{contract_address}.json")
    with open(filename, 'w') as file:
        json.dump({'summary': summary}, file, indent=4)

def process_sites(data, sites_without_summary):
    schema = {
        "properties": {
            "news_article_title": {"type": "string"},
            "news_article_summary": {"type": "string"},
            "email": {"type": "string"}
        },
        "required": ["news_article_title", "news_article_summary", "email"],
    }

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    for site, contract in sites_without_summary:
        try:
            docs_transformed, telegram_links, email = extract_content(site)
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=0)
            splits = splitter.split_documents(docs_transformed)
            if not splits or splits[0].page_content == "":
                targetSummary = f"Failed to extract content for site {site}"
            else:
                try:
                    extracted_content = create_extraction_chain(schema=schema, llm=llm).run(splits[0].page_content)
                    combined_content = [f"{item.get('news_article_title', '')} - {item.get('news_article_summary', '')}\n\n" for item in extracted_content]
                    targetSummary = ' '.join(combined_content)
                except json.JSONDecodeError as e:
                    targetSummary = f"Failed to extract content due to JSON decoding error: {str(e)}"
                except Exception as e:
                    targetSummary = f"Unexpected error during extraction: {str(e)}"

            print(f"Summary for {contract} {site}: {targetSummary}")
            save_summary_and_proposal(contract, targetSummary)

            # Update the data list to mark the site as processed
            for entry in data:
                if entry.get('web_domains') and entry['web_domains'][0] == site:
                    # Combine existing and new telegram groups, ensuring no duplicates
                    existing_telegram_groups = set(entry.get('telegram_groups', []))
                    telegram_links = list(existing_telegram_groups.union(telegram_links))
                    entry['telegram_groups'] = telegram_links

                    # Combine existing and new email addresses, ensuring no duplicates
                    existing_emails = set(entry.get('emails', []))
                    existing_emails.add(email)
                    if email != None and len(email) > 0 and len(email) < 60:
                        entry['emails'] = list(existing_emails)
                    entry['p6'] = True       
        except Exception as e:
            print(f"Failed to process site {site} due to error: {str(e)}")



def save_updated_data(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def main():
    data = load_data_from_file(MAINFILE)
    sites_without_summary = filter_sites_without_summary(data)
    print(f"Found {len(sites_without_summary)} sites without summary")
    sites_without_summary = sites_without_summary[:6]
    if not sites_without_summary:
        print("All sites processed")
    else:
        process_sites(data, sites_without_summary)
        save_updated_data(MAINFILE, data)


if __name__ == "__main__":
    main()
