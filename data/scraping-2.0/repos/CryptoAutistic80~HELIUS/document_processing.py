import os
import json
import re
import glob
import PyPDF2
import spacy
import nltk
import uuid
import openai
import pinecone
import numpy as np
from config import OPENAI_KEY, PINECONE_KEY
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import html2text
from readability import Document

nltk.download('punkt')

pdf_path = 'Doc_Processing/PDF_documents/'
output_path = 'Doc_Processing/embeddings/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

processed_docs_path = os.path.join('Doc_Processing')
if not os.path.exists(processed_docs_path):
    os.makedirs(processed_docs_path)

nlp = spacy.load('en_core_web_sm')

def gpt_embed(text):
    content = text.encode(encoding='ASCII', errors='ignore').decode()  # fix any UNICODE errors
    openai.api_key = OPENAI_KEY
    response = openai.Embedding.create(
        input=content,
        engine='text-embedding-ada-002'
    )
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

def extract_text(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        pages = []
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            pages.append({'page_num': page_num+1, 'text': page.extract_text()})
        return pages

def split_text(text):
    sentences = nltk.sent_tokenize(text)
    return ['\n'.join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]

def clean_text(text):
    # Remove unwanted Unicode characters (e.g., \u2022)
    text = text.encode('ascii', errors='ignore').decode('ascii')
    # Replace newline characters with spaces
    text = text.replace('\n', ' ')
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_file(file):
    pages = extract_text(file)

    embeddings = []
    for page in pages:
        page_num = page['page_num']
        text = page['text']
        page_groups = split_text(text)
        for i, group in enumerate(page_groups):
            output_uuid = uuid.uuid4()
            output_filename = os.path.join(output_path, f'{output_uuid}.json')
            cleaned_text = clean_text(group)
            metadata = {'filename': os.path.basename(file), 'file_number': i+1, 'page_number': page_num, 'uuid': str(output_uuid), 'text': cleaned_text}
            vector = gpt_embed(cleaned_text)
            vector_np = np.array(vector)  # Convert the list to a NumPy array
            embeddings.append((metadata['uuid'], vector_np))

            # Save the metadata and cleaned text to a JSON file
            with open(output_filename, 'w') as f:
                json.dump(metadata, f, indent=4)

    # Upsert embeddings to Pinecone in smaller batches
    batch_size = 100
    pinecone.init(api_key=PINECONE_KEY, environment='us-east1-gcp')
    pinecone_indexer = pinecone.Index("core-69")
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]
        pinecone_indexer.upsert([(unique_id, vector_np.tolist()) for unique_id, vector_np in batch], namespace="knowledge")


def process_files():
    processed_files = []
    processed_files_filename = os.path.join(processed_docs_path, 'processed_files.json')
    if os.path.exists(processed_files_filename):
        with open(processed_files_filename, 'r') as f:
            processed_files = json.load(f)

    for file in [file for file in glob.glob(os.path.join(pdf_path, '*.pdf')) if os.path.basename(file) not in processed_files]:
        process_file(file)
        print(f"Processed PDF file: {file}")

        # Add the processed file to the list of processed files
        processed_files.append(os.path.basename(file))

        # Save the list of processed files
        with open(processed_files_filename, 'w') as f:
            json.dump(processed_files, f, indent=4)

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_all_links(url, soup):
    links = set()
    domain = urlparse(url).scheme + "://" + urlparse(url).hostname

    for a_tag in soup.find_all("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            continue

        href = urljoin(url, href)

        parsed_href = urlparse(href)
        if parsed_href.scheme and parsed_href.hostname and parsed_href.path:
            href = parsed_href.scheme + "://" + parsed_href.hostname + parsed_href.path
        else:
            continue

        if not is_valid_url(href):
            continue

        if domain in href:
            links.add(href)

    return links


def get_title_text(url):
    r = requests.get(url)

    # Extract the main content using the readability package
    document = Document(r.text)
    title = document.title()
    main_content = document.summary()
    main_soup = BeautifulSoup(main_content, "html.parser")

    # Convert the main content area to text using html2text
    html_string = str(main_soup) if main_soup else ""
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = False  # Include links in the output
    text_maker.ignore_images = True
    text_maker.ignore_tables = True
    text_maker.ignore_anchors = True
    text = text_maker.handle(html_string).strip()

    # Include only specific links (Twitter, Discord, Medium, and GitHub)
    for a_tag in main_soup.find_all("a"):
        href = a_tag.attrs.get("href")
        if href and any(domain in href for domain in ["twitter.com", "discord.gg", "medium.com", "github.com"]):
            text += f"\n{a_tag.text}: {href}"

    text = clean_git_text(text)

    return title, text


def crawl(url, data, max_depth=1, depth=0):
    if depth > max_depth:
        return

    print(f"Processing URL: {url}")  # Add print statement here
    title, text = get_title_text(url)
    data[url] = {'title': title, 'text': text}

    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    sub_urls = get_all_links(url, soup)

    for sub_url in sub_urls:
        if sub_url not in data:
            print(f"Crawling sub URL: {sub_url}")
            crawl(sub_url, data, max_depth=max_depth, depth=depth + 1)

def clean_git_text(text):
    text = re.sub(r'\^K', '', text)  # Remove ^K characters
    allowed_domains = ['twitter.com', 'discord.com', 'discord.gg', 'medium.com', 'github.com']

    # Keep only the desired links
    for match in re.finditer(r'\[(.+?)\]\((.+?)\)', text):
        label, url = match.groups()
        if not any(allowed_domain in url for allowed_domain in allowed_domains):
            text = text.replace(match.group(0), label)

    return text

def process_crawled_data(url, data):
    title, text = data['title'], data['text']
    groups = split_text(text)

    embeddings = []
    for i, group in enumerate(groups):
        output_uuid = uuid.uuid4()
        output_filename = os.path.join(output_path, f'{output_uuid}.json')
        cleaned_text = clean_git_text(group)
        metadata = {'url': url, 'title': title, 'group_number': i+1, 'uuid': str(output_uuid), 'text': cleaned_text}
        vector = gpt_embed(cleaned_text)
        vector_np = np.array(vector)  # Convert the list to a NumPy array
        embeddings.append((metadata['uuid'], vector_np))

        # Save the metadata and cleaned text to a JSON file
        with open(output_filename, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Processed group {i + 1} of {len(groups)} for URL: {url}")

    # Upsert embeddings to Pinecone in smaller batches
    batch_size = 100
    pinecone.init(api_key=PINECONE_KEY, environment='us-east1-gcp')
    pinecone_indexer = pinecone.Index("core-69")
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]
        pinecone_indexer.upsert([(unique_id, vector_np.tolist()) for unique_id, vector_np in batch], namespace="knowledge")


def process_gitbook_urls():
    gitbook_urls = []
    gitbook_urls_filename = os.path.join(processed_docs_path, 'GitBook_URL.txt')
    if os.path.exists(gitbook_urls_filename):
        with open(gitbook_urls_filename, 'r') as f:
            gitbook_urls = [url.strip() for url in f.readlines()]

    processed_urls = []
    processed_urls_filename = os.path.join(processed_docs_path, 'Processed_URLs.json')
    if os.path.exists(processed_urls_filename):
        with open(processed_urls_filename, 'r') as f:
            processed_urls = json.load(f)

    for url in [url for url in gitbook_urls if url not in processed_urls]:
        data = {}
        crawl(url, data, max_depth=5)  # You can set max_depth to any desired depth level

        for crawled_url, crawled_data in data.items():
            process_crawled_data(crawled_url, crawled_data)
            print(f"Processed GitBook URL: {crawled_url}")

            # Add the processed URL to the list of processed URLs
            processed_urls.append(crawled_url)

        # Save the list of processed URLs
        with open(processed_urls_filename, 'w') as f:
            json.dump(processed_urls, f, indent=4)



def process_all():
    process_files()
    process_gitbook_urls()


if __name__ == "__main__":
    process_all()
    print("Document processing completed successfully.")