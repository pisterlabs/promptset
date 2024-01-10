from atlassian import Confluence
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import Prompt
import streamlit as st
import os

confluence = Confluence(
    url=st.secrets["CONFLUENCE_URL"],
    username=st.secrets["CONFLUENCE_USERNAME"],
    password=st.secrets["CONFLUENCE_PASSWORD"]
    )
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


pre_prompt = """
You are being given a question that is meant to be answered by searching the 
Keboola Confluence docs. The only way to answer this question is to search the docs. 
In order to search the docs, you must generate a CQL query that will 
return the most relevant results. 
Confluence uses Apache Lucene for text indexing, 
which provides a rich query language. 
Much of the information about how to generate a CQL query 
is derived from the Query Parser Syntax
page of the Lucene documentation.


A query is broken up into terms and operators. 
There are two types of terms: Single Terms and Phrases.

A Single Term is a single word such as "test" or "hello".

A Phrase is a group of words surrounded by double quotes such as "hello dolly".

Note: All query terms in Confluence are case insensitive.

Your task is to take the input question and generate a CQL query that will return the most relevant results.
"""

def generate_cql_query_keywords(input_text):
    # create a more advanced function to generate a CQL query. 
    # It should utlize an LLM to extract keywords from the input 
    # text and then use those to power the query
        llm = OpenAI()
        cql_query = llm.predict(pre_prompt + input_text)
        return f"text ~ '{cql_query}'"


def conflu_search(input_text: str) -> GPTVectorStoreIndex:
    st.write('Generating CQL query...')
    st.progress(0.1)
    #query_raw = generate_cql_query_raw(input_text)
    query_keywords = generate_cql_query_keywords(input_text)
    st.write('Query generated!')
    st.progress(0.2)
    st.write('Searching Confluence...')
    st.progress(0.3)
    pages_keywords = confluence.cql(query_keywords)
    pages = pages_keywords
    documents = []

    progress = 0.4
    for page in pages['results']:
        st.write(f"Found a potentially revevant page: {page['content']['title']}")
        progress += 0.01
        st.progress(progress)
        # Check the local directory to see if we already have the page's content
        if os.path.exists(f"data/{page['content']['id']}.txt"):
            st.write(f"Found the page's content locally: {page['content']['title']}")
            with open(f"data/{page['content']['id']}.txt", "r") as f:
                documents.append(f.read())
                f.close()
            continue
        
        # If we don't have the page's content, then get it from Confluence
        else:
            st.write(f"Getting the page's content from Confluence: {page['content']['title']}")
            content = confluence.get_page_by_id(page['content']['id'], expand='body.view')
            documents.append(content['body']['view']['value'])
             # add each page's content as a txt file in the data directory
            with open(f"data/{page['content']['id']}.txt", "w") as f:
                f.write(content['body']['view']['value'])
                f.close()
    st.write('Finished searching Confluence!')
        # convert documents to a string
    documents = SimpleDirectoryReader('data').load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    st.write('Index created!')
    

    return index

if __name__ == "__main__":
    conflu_search()