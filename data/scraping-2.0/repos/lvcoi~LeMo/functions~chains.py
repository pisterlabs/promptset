import os
import requests
import json
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import find_dotenv, load_dotenv


# load env variables
load_dotenv(find_dotenv())
# Two models are available: fast and smart 
fast = "gpt-3.5-turbo"
smart = "gpt-4-0613"

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# serp request to get list of news

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': SERPAPI_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()

    print("search results: ", response_data)
    return response_data


# llm to choose the best statutes or case law
def find_best_article_urls(response_data, query):
    # turn json into string
    response_str = json.dumps(response_data)

    # create llm to choose best statutes or case law
    llm = OpenAI(model_name=fast, temperature=0) # type: ignore
    template = """
    You are a world class paralegal & researcher, you are extremely good at find most relevant case law to certain topic;
    {response_str}
    Above is the list of search results for the query {query}.
    Please choose the best 3 statutes or case law from the list, return ONLY an array of the urls, do not include anything else; return ONLY an array of the urls, do not include anything else
    """

    # Commit response_str to memory
    
    prompt_template = PromptTemplate(
        input_variables=["response_str", "query"], template=template)

    article_picker_chain = LLMChain(
        llm=llm, prompt=prompt_template, verbose=True)

    urls = article_picker_chain.predict(response_str=response_str, query=query)

    # Convert string to list
    url_list = json.loads(urls)
    print(url_list)

    return url_list

# get content from each article & create a vector database
def get_content_from_urls(urls):   
    # use unstructuredURLLoader
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    return data

def summarize(data, query):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=3000, chunk_overlap=400, length_function=len)
    text = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings() # type: ignore

    db = FAISS.from_documents(text, embeddings)
    db.save_local("faiss_index")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(query)

    llm = OpenAI(model_name=fast, temperature=.2) # type: ignore
    template = """
    {docs}
    You are a world class paralegal & research assistant, text above is some context about {query}
    Please write seperate, clear and conscice To Do list for both the client and the attorney about {query} using the text above, and following all rules below:
    1/ The list needs to be engaging, informative with good data
    2/ The list needs to be around 7-10 bulleted items
    3/ The list needs to address the {query} topic very well
    4/ The list needs to have links to the source data
    5/ The list needs to be written in a way that is easy to read and understand
    6/ The list needs to give audience actionable advice & insights too

    SUMMARY:
    """

    prompt_template = PromptTemplate(input_variables=["text", "query"], template=template)

    summarizer_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    summaries = []
    
    for chunk in enumerate(text):
        summary = summarizer_chain.predict(text=chunk, query=query)
        summaries.append(summary)

    print(summaries)
    return summaries

# Turn summarization into client and attorney To Do list
def generate_list(summaries, query):
    summaries_str = str(summaries)

    llm = OpenAI(model_name=fast, temperature=0.4) # type: ignore
    template = """
    {summaries_str}

    You are a world class paralegal & research assistant, text above is some context about {query}
    Please write seperate, clear and conscice To Do list for both the client and the attorney about {query} using the text above, and following all rules below:
    1/ The list needs to be engaging, informative with good data
    2/ The list needs to be around 7-10 bulleted items
    3/ The list needs to address the {query} topic very well
    4/ The list needs to have links to the source data
    5/ The list needs to be written in a way that is easy to read and understand
    6/ The list needs to give audience actionable advice & insights too

    Client and Attorney ToDo_ClientAttorney list:
    """

    prompt_template = PromptTemplate(input_variables=["summaries_str", "query"], template=template)
    DoChain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    ToDo_ClientAttorney = DoChain.predict(summaries_str=summaries_str, query=query)

    return ToDo_ClientAttorney

# 