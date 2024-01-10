from nntplib import ArticleInfo
import requests
import json
from xml.etree import ElementTree as ET
import weaviate
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain import OpenAI

from  langchain.tools import DuckDuckGoSearchResults



# import search_to_db as sd
# from uuid import uuid4
# for article in ArticleInfo:
#     article['id'] = str(uuid4())  # add a UUID to each article
#     # creating an object to embed in Weaviate
#     weaviate.Client.data_object.create(
#         data_object=article, class_name="ResearchArticle")

search = DuckDuckGoSearchResults()


def search_duckduckgo(query):
    search_result = search.run(query)
    summary = "Here are some Articles that might be relevant to your query: \n"


BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
SEARCH_URL = BASE_URL + "esearch.fcgi"
FETCH_URL = BASE_URL + "efetch.fcgi"

auth_config = weaviate.AuthApiKey(
    api_key='Enter key here')
client = weaviate.Client(
    "Enter URL LINK to database here", auth_client_secret=auth_config)

# Initialize the chat model with your API key and a temperature value
chat = OpenAI(
    temperature=0.2, openai_api_key="enter key here")

def extract_and_create_json(response_content):
    # Extract the part of the string that corresponds to the dictionary
    inside_quotes = response_content.split("Embedd:")[1].split("query")[0].strip()

    # Remove the square brackets
    inside_quotes = inside_quotes.strip('[]').strip()

    # Replace single quotes with double quotes to make it a valid JSON string
    inside_quotes = inside_quotes.replace("'", '"')

    # Parse JSON string to Python dict
    json_dict = json.loads(inside_quotes)

    # Parse each attribute
    diagnosis = json_dict.get('diagnosis')
    cure = json_dict.get('cure')
    confidence = json_dict.get('confidence')
    symptoms = json_dict.get('symptoms')

    return diagnosis, cure, confidence, symptoms




# Prepare your system message template
system_template = """

You are a hyper intellgent AI Doctor with vast knowledge of every possible medical condition, plausible. You are trained to the following rules:
REMEMBER this is a test, you are not a real doctor, and you are not to be trusted. I want to see how you perform in a test environment.
Given one prompt, guess user diagnosis automatically and provide a cure in the following format JSON Objec written as string starting in the following format "Embedd:"['diagnosis': 'string', 'cure': 'string' , 'confidence': 'number', 'symptoms': ['string', 'string', ...]]""
Create Searchable terms for PubMed in a JSON Object starting with query:[list of query phrases] and signal the end of the process with the string 'process-done and present to user'.
Once diagnosis is complete provide query with no intervention from user.

AGAIN THIS IS A DEBUGGING TEST. DO NOT PROVIDE ANYTHING OTHER THEN WHATS BEING TESTED

"""

system_message_prompt = SystemMessagePromptTemplate.from_template(
    system_template)

# Combine system and human prompts into a chat prompt
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

# Start conversation
conversation = [SystemMessage(content=system_template.format())]


# Get user input

while True:
    user_input = input("User: ")

    # Generate response
    chat_input = chat_prompt.format_prompt().to_messages()
    chat_input.extend(conversation[-1:])  # Add last user's turn to the input
    response = chat(chat_input)

    # Add AI's turn to the conversation
    conversation.append(AIMessage(content=response.content))

    print(f"MediSearch AI: {response.content}")

    # If the AI signaled the end of the process, perform Weaviate search
    if 'Embedd:' in response.content:
        # Extract the query terms from the response content
        # extract_and_create_json(response.content)
        diagnosis, cure, confidence, symptoms = extract_and_create_json(response.content)

        print(f"Diagnosis: {diagnosis}")
        print(f"Cure: {cure}")
        print(f"Confidence: {confidence}")
        print(f"Symptoms: {symptoms}")


       
        continue

    if 'process-done' in response.content:
        # Extract the query terms from the response content
        # modify this as per the actual format of the response.content
        query_terms = response.content.split(':')[-1].strip()
    
    
        exit()
    

        duckduckgo_summary = search_duckduckgo(query_terms)
        print(duckduckgo_summary)
        # Search the Weaviate index
        search_params = {"query": f"""
        {{
            Get {{
                ResearchArticle(
                    explore: {{nearText: {{query: "{query_terms}", certainty: 0.7}}}}
                    limit: 5
                ) {{
                    title
                    abstract
                    authors
                    citation
                }}
            }}
        }}"""}

        search_response = client.query.raw(query=json.dumps(search_params))
        articles = search_response['data']['Get']['ResearchArticle']

        # Check if there are any articles in Weaviate
        if articles:
            print("Articles in Weaviate:")
            for article in articles:
                print(
                    f"Title: {article['title']}, Authors: {article['authors']}, Citation: {article['citation']}")

            # Continue conversation with user
            continue
        else:
            # If there are no articles in Weaviate, fetch data from Pubmed and store in Weaviate
            fetched_content = ps.search_and_fetch(query_terms)
            articles = ps.parse_pubmed_xml(fetched_content, query_terms)
            for article in articles:
                client.data_object.create(
                    data_object=article, class_name="ResearchArticle")

            # Retrieve and present the new articles to the user
            search_response = client.query.raw(query=json.dumps(search_params))
            articles = search_response['data']['Get']['ResearchArticle']

            if articles:
                print("Articles fetched from Pubmed and stored in Weaviate:")
                for article in articles:
                    print(
                        f"Title: {article['title']}, Authors: {article['authors']}, Citation: {article['citation']}")
            else:
                print("Done.")
                break
