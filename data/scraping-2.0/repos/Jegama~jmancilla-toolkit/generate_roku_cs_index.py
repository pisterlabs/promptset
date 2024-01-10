from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    LangchainEmbedding,
    GPTVectorStoreIndex
)
# from llama_index.indices.document_summary import GPTDocumentSummaryIndex
# from langchain.llms import AzureOpenAI
# from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

from llama_index.node_parser import SimpleNodeParser
from playwright.sync_api import sync_playwright

import json, time, openai, os, re, requests

from dotenv import load_dotenv
load_dotenv()

# if temp forlder doesn't exist, create it
if not os.path.exists('temp'):
    os.makedirs('temp')

def get_error_codes(document):
    # get first line of file temp/temp.txt
    with open('temp/temp.txt', 'r', encoding='utf-8') as f:
        title = f.readline()

    new_doc = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a meticulous librarian."},
            {"role": "user", "content": f"From the following text, please extract all the error codes and their descriptions, as well as the solutions. Please make sure you return all the numbers of the error codes.\n\nTitle:{title}\n{document.text}"}
        ]
    )
    response = f"Title:{title}\n{new_doc['choices'][0]['message']['content']}"
    # write response into a temp file
    with open('temp/temp.txt', 'w', encoding='utf-8') as f:
        f.write(response)

    return SimpleDirectoryReader('temp').load_data()

def extract_text_from_div(url, class_name):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()

        page.goto(url, timeout=0)
        page.wait_for_load_state('networkidle')

        div_element = page.query_selector(f".{class_name}")
        if div_element:
            text_content = div_element.inner_text()
            with open('temp/temp.txt', 'w', encoding='utf-8') as f:
                f.write(text_content)
        else:
            print(f"No element found with class: {class_name}")

        browser.close()

    return SimpleDirectoryReader('temp').load_data()

def find_urls_in_webpage(url):
    # Fetch the web page content
    response = requests.get(url)
    webpage_content = response.text

    # Define the regular expression pattern
    pattern = r'https:\/\/support\.roku\.com\/article\/\d+'

    # Find all URLs matching the pattern
    urls = re.findall(pattern, webpage_content)

    return urls

urls = find_urls_in_webpage('https://support.roku.com/sitemap.xml')

# remove duplicates from list
urls = list(dict.fromkeys(urls))
print(f'\nFound {len(urls)} unique urls')

parser = SimpleNodeParser()

# Azure OpenAI service Context
# llm = LLMPredictor(llm=AzureOpenAI(temperature=1, deployment_name='gpt-35-turbo', model_name='gpt-35-turbo'))
# embeddings = LangchainEmbedding(OpenAIEmbeddings(deployment="text-embedding-ada-002"))
# service_context = ServiceContext.from_defaults(llm_predictor=llm, chunk_size_limit=1024, embed_model=embeddings)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=1024)

cs_index = GPTVectorStoreIndex([], service_context=service_context)
error_codes_index = GPTVectorStoreIndex([], service_context=service_context)

docid_to_url = {}
tokens_used = 0
missing_urls = []

start = time.time()

def populate_index(index, index2, documents):
    nodes = parser.get_nodes_from_documents(documents)
    docid_to_url[nodes[0].doc_id] = page
    index.insert_nodes(nodes)

    # if the documents includes any mention of error codes using regex
    if re.search(r'error code', documents[0].text, re.IGNORECASE):
        print(f'Found error codes in {page}')
        error_codes = get_error_codes(documents[0])
        nodes_error_codes = parser.get_nodes_from_documents(error_codes)
        docid_to_url[nodes_error_codes[0].doc_id] = page
        index2.insert_nodes(nodes_error_codes)

print('\nPopulating index...')
for page in urls:
    documents = extract_text_from_div(page, 'article-content-wrapper')
    try:
        with open('temp/temp.txt', 'r', encoding='utf-8') as f:
            title = f.readline()
        print(f'\nProcessing {page} - {title}')
        populate_index(cs_index, error_codes_index, documents)                
        os.remove('temp/temp.txt')
    except:
        print(f'Error processing {page}')
        missing_urls.append(page)

for page in missing_urls:
    documents = extract_text_from_div(page, 'article-content-wrapper')
    try:
        with open('temp/temp.txt', 'r', encoding='utf-8') as f:
            title = f.readline()
        print(f'\nProcessing {page} - {title}')
        populate_index(cs_index, error_codes_index, documents)                
        os.remove('temp/temp.txt')
    except:
        print(f'Error processing {page}')
        pass

print(f'\nIndex populated in {(time.time() - start)/60} minutes')

total_cost = (cs_index._service_context.embed_model._total_tokens_used/1000) * 0.0004
total_cost += (error_codes_index._service_context.embed_model._total_tokens_used/1000) * 0.0004
print('\nTotal cost: $', total_cost)

cs_index.storage_context.persist(persist_dir='cs_index')
error_codes_index.storage_context.persist(persist_dir='error_codes_index')

with open('cs_docid_to_url.json', 'w') as f:
    json.dump(docid_to_url, f)