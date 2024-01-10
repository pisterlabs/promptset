from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader, PromptHelper, MockLLMPredictor, LLMPredictor, ServiceContext
from langchain import OpenAI
from IPython.display import Markdown, display
import wikipediaapi
import streamlit as st
from googleapiclient.discovery import build
import os

# Set api keys for openai api and google custom search engine api
os.environ['OPENAI_API_KEY'] = st.secrets["openai"]
my_api_key = st.secrets["my_api_key"]
my_cse_id = st.secrets["my_cse_id"]

# Connecting to the wikipedia api
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)


#Create folders data, index, to_be_embedded

if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists('index'):
    os.mkdir('index')
if not os.path.exists('to_be_embedded'):
    os.mkdir('to_be_embedded')


# Define LLM predictor openai_api_key="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"
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.1, model_name="gpt-3.5-turbo"))

# Set max input size
max_input_size = 4096 
# Set no of output tokens
num_output = 512
# Set max chunk overlap
max_chunk_overlap = 20
# Set chunk size limit
chunk_size_limit = 600

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

# Function to perform a google search on the user input question
def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']

# Extract snippet from the json response of the google custom search api
def extract_snippets(results):
    all_snippets = ""
    
    for result in results:
        snippet = result.get('snippet', 'No snippet available')
        all_snippets += snippet + " "

    return all_snippets.strip()




def wiki_search(subject_area):
    # Search the data folder and make a list of all the files and strip the .json extension from it
    files = os.listdir('data')
    files_json = [file.replace('.json', '') for file in files]
    # Get page from wikipedia using user input
    page_py = wiki_wiki.page(subject_area)
    title = page_py.title
    # Convert title to lowercase
    title =  title.lower()

    # Search the list of files for the user input expertise area. Does the user input exist in list of files (files_json)
    if title in files_json:
        # if found then load the index
        index = GPTSimpleVectorIndex.load_from_disk(f'index/{title}.json')
    
    elif title not in files_json:
        # if not found 
        # Get the knowledge from wikipedia and save the text to a file
        text = page_py.text
        word_count = len(text.split())

        # Create a json file with the title and utf-8 encoding
        with open(f"data/{title}.json", "w", encoding="utf-8") as f:
            f.write(text)
        
        # Save the file to to_be_embedded folder
        with open(f"to_be_embedded/{title}.json", "w", encoding="utf-8") as f:
            f.write(text)
        

        # Perform indexing i.e Embed the text in the index

        # Check if the index exists
        if os.path.exists(f'index/{text}.json'):
            index = GPTSimpleVectorIndex.load_from_disk(f'index/{title}.json')
        else:
            documents = SimpleDirectoryReader('to_be_embedded').load_data()
            print(len(documents))
            service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
            index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

            index.save_to_disk(f'index/{title}.json')

        # Indexing completed so clear the to_be_embedded folder
        files = os.listdir('to_be_embedded')
        for file in files:
            os.remove(f'to_be_embedded/{file}')
    return index


# Query the index and return the result
def ssearch(subject_area, question):
    index = wiki_search(subject_area)
    
    query = question
    # If query exceeds the maximum length allowed by google custom search api, break it into smaller, more focused queries.
    if len(query) > 2048:
        words = query.split()
        current_query = ""

        # Initialize an empty list for queries
        queries = []  

        for word in words:
            if len(current_query) + len(word) + 1 <= 2048:
                current_query += f" {word}"
            else:
                queries.append(current_query.strip())
                current_query = word

        queries.append(current_query.strip())

    results = google_search(query, my_api_key, my_cse_id, num=10)

    # Search results from google aren't really that helpfull so we are just using it to provide context to chatgpt
    context = extract_snippets(results)
    
    response = index.query(question)

    return response.response + " " + context

