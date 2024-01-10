from atlassian import Confluence, errors
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex
from langchain.llms import OpenAI
import os
import sys
from decouple import config
from typing import Union
import logging
import uuid

confluence = Confluence(
    url=config("CONFLUENCE_URL"),
    username=config("CONFLUENCE_USERNAME"),
    password=config("CONFLUENCE_PASSWORD")
)

try:
    main_file_path = '~'
except Exception as e:
    print("Error getting main_file_path:", str(e))
    main_file_path = ''

print("main_file_path:", main_file_path)

data_dir = os.path.join(os.path.dirname(os.path.abspath(main_file_path)), "data")
print("data_dir:", data_dir)
pre_prompt = """
You are being given a question that needs to be answered by searching the Keboola Confluence docs. To find the answer, 
you must generate a CQL query that will return the most relevant results.

Confluence uses Apache Lucene for text indexing, which provides a rich query language. Much of the information about how
 to generate a CQL query can be found in the Query Parser Syntax page of the Lucene documentation.

A query is composed of terms and operators. There are two types of terms: single terms and phrases.

- Single terms: These are single words like "test" or "hello".
- Phrases: These are groups of words surrounded by double quotes, such as "hello dolly".

Remember that all query terms in Confluence are case insensitive.

Your task is to take the input question and generate a CQL query that will return the most relevant results. Focus on 
using shorter terms rather than long ones, and avoid including phrases like "step by step guide" or "exact process."
Do not include words like "step" or "guide" in the search query.


Please respond with a valid Atlassian CQL query that you believe will yield the most relevant results.

Examples:
text ~ "Keboola AND Python AND component"
text ~ "BYODB AND process"
text ~ "Keboola AND component AND publish"
"""

pre_prompt_history = """
You are being given a conversation history between a ChatGPT bot and a user asking a question
that needs to be answered by searching the Keboola Confluence docs. To find the answer, 
you must generate a CQL query that will return the most relevant results.

Confluence uses Apache Lucene for text indexing, which provides a rich query language. Much of the information about how
 to generate a CQL query can be found in the Query Parser Syntax page of the Lucene documentation.

A query is composed of terms and operators. There are two types of terms: single terms and phrases.

- Single terms: These are single words like "test" or "hello".
- Phrases: These are groups of words surrounded by double quotes, such as "hello dolly".

Remember that all query terms in Confluence are case insensitive.

Your task is to take the input question and generate a CQL query that will return the most relevant results. Focus on 
using shorter terms rather than long ones, and avoid including phrases like "step by step guide" or "exact process."
Do not include words like "step" or "guide" in the search query.

Please respond with a valid Atlassian CQL query that you believe will yield the most relevant results.

Examples:
text ~ "Keboola AND Python AND component"
text ~ "BYODB AND process"
text ~ "Keboola AND component AND publish"
"""


def create_unique_folder():
    folder_name = str(uuid.uuid4())
    folder_path = os.path.join(data_dir, folder_name)
    os.makedirs(folder_path)
    return folder_path


def generate_cql_query_keywords(input_text: str, user_messages: list = None, bot_messages: list = None) -> str:
    llm = OpenAI()

    if user_messages and bot_messages:
        prompt = pre_prompt_history + f"bot messages: {bot_messages}, user_messages: {user_messages}"
    else:
        prompt = pre_prompt + input_text

    logging.info(f"Prompting: {prompt}")
    response = llm.predict(prompt)
    cql_query = response.replace("\n", "").strip(" ")
    return cql_query


def query_conflu(cql_query: str):
    logging.info(f"Query: {cql_query}")

    pages = None
    try:
        pages = confluence.cql(cql_query)
    except errors.ApiValueError:
        logging.error(f"Query: {cql_query} is invalid.")

    return pages


def download_documents(pages: list):
    documents = []
    query_directory = create_unique_folder()

    for page in pages:
        # Check the local directory to see if we already have the page's content
        if os.path.exists(f"{query_directory}/{page['content']['id']}.txt"):
            with open(f"{query_directory}/{page['content']['id']}.txt", "r") as f:
                documents.append(f.read())
                f.close()
            continue

        # If we don't have the page's content, then get it from Confluence
        else:
            content = confluence.get_page_by_id(page['content']['id'], expand='body.view')
            documents.append(content['body']['view']['value'])
            # add each page's content as a txt file in the data directory
            with open(f"{query_directory}/{page['content']['id']}.txt", "w") as f:
                f.write(content['body']['view']['value'])
                f.close()

    # convert documents to a string
    logging.info(f"Using document directory: {query_directory}")
    documents = SimpleDirectoryReader(f"{query_directory}").load_data()
    index = GPTVectorStoreIndex.from_documents(documents)

    return index


def conflu_search(
        search: str,
        user_messages: list = None,
        bot_messages: list = None) -> Union[GPTVectorStoreIndex, None]:
    query_counter = 0
    while query_counter < 3:
        query_counter += 1
        query = generate_cql_query_keywords(search, user_messages, bot_messages)
        r = query_conflu(query)
        if r is not None and r.get("results"):
            index = download_documents(r.get("results"))
            return index
    return None


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
    conflu_search("What is the complete BYODB process?")
