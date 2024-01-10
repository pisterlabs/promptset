import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import json
from serpapi import GoogleSearch
from autogen import config_list_from_json
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen import UserProxyAgent
import autogen
import arxiv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter

from agent_tools import search_youtube, get_youtube_transcript


load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serp_api_key = os.getenv("SERP_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
airtable_api_key = os.getenv("AIRTABLE_API_KEY")
config_list = config_list_from_json("OAI_CONFIG_LIST")


client = arxiv.Client()
# https://pypi.org/project/arxiv/

# ------------------ Create functions ------------------ #

# def search_arxiv(query):
#     search = arxiv.Search(
#         query = query,
#         max_results = 10,
#         sort_by = arxiv.SortCriterion.SubmittedDate
#         )
#     results = client.results(search)
#     for r in client.results(search):
#         print(r.title)
#     all_results = list(results)
#     print([r.title for r in all_results])
#     return all_results

# def download_paper(paper_id):
#     paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
#     # Download the PDF to the PWD with a default filename.
#     paper.download_pdf()
#     # Download the PDF to the PWD with a custom filename.
#     paper.download_pdf(filename=f"{paper_id}.pdf")
#     # Download the PDF to a specified directory with a custom filename.
#     paper.download_pdf(dirpath="./mydir", filename=f"{paper_id}.pdf")

def search_youtube(query):
    number_of_results = int(5)

    try:
        response = requests.get('https://serpapi.com/search.json', params={
            'engine': 'youtube',
            'search_query': query,
            'api_key': serp_api_key,
        })

        if response.status_code != 200:
            raise Exception(f"HTTP error occurred: {response.status_code}")

        data = response.json()
        video_results = data.get('video_results', [])[:number_of_results]
        enhanced_video_results = []
        for video in video_results:
            video_id = video['link'].split('v=')[-1]
            transcript = get_youtube_transcript(video_id)
            video['transcript'] = transcript
            enhanced_video_results.append(video)
        data['video_results'] = enhanced_video_results
        return enhanced_video_results
    except Exception as err:
        print(f"An error occurred: {err}")

    return data

def get_youtube_transcript(video_id):
    print("Getting transcript for video:", video_id)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ' '.join(segment['text'] for segment in transcript)
        return full_transcript
    except Exception as e:
        print(f"An error occurred while fetching the transcript: {e}")
        return None

def search_google_scholar(query):
    try:
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": serp_api_key,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])
        return organic_results
    except Exception as err:
        print(f"An error occurred: {err}")

# Function for google search
def google_search(query):    
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    # print("RESPONSE:", response.text)
    return response.text

# Function for scraping
def summary(objective, content):
    llm = ChatOpenAI(temperature = 0, model = "gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size = 10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])
    
    summary_chain = load_summarize_chain(
        llm=llm, 
        chain_type='map_reduce',
        map_prompt = map_prompt_template,
        combine_prompt = map_prompt_template,
        verbose = False
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

def web_scraping(objective: str, url: str):
    #scrape website, and also will summarize the content based on objective if the content is too large
    #objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url        
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(f"https://chrome.browserless.io/content?token={browserless_api_key}", headers=headers, data=data_json)
    
    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        # print("CONTENTTTTTT:", text)
        if len(text) > 10000:
            output = summary(objective,text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")        


# Function for get airtable records
def get_airtable_records(base_id, table_id):
    url = f"https://api.airtable.com/v0/{base_id}/{table_id}"

    headers = {
        'Authorization': f'Bearer {airtable_api_key}',
    }

    response = requests.request("GET", url, headers=headers)
    data = response.json()
    print(data)
    return data


# Function for update airtable records

def update_single_airtable_record(base_id, table_id, id, fields):
    url = f"https://api.airtable.com/v0/{base_id}/{table_id}"

    headers = {
        'Authorization': f'Bearer {airtable_api_key}',
        "Content-Type": "application/json"
    }

    data = {
        "records": [{
            "id": id,
            "fields": fields
        }]
    }

    response = requests.patch(url, headers=headers, data=json.dumps(data))
    data = response.json()
    return data


# ------------------ Create agent ------------------ #

# Create user proxy agent
user_proxy = UserProxyAgent(name="user_proxy",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    system_message=" A human admin. Interact with the Director to discuss the plan. Plan execution needs to be approved by this admin.",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=1
    )


# Create researcher agent
researcher = GPTAssistantAgent(
    name = "researcher",
    llm_config = {
        "config_list": config_list,
        "assistant_id": "asst_tHMkLHXS8JrK1xG8FN6GzT5t"
    }
)

researcher.register_function(
    function_map={
        "web_scraping": web_scraping,
        "google_search": google_search,
        "search_google_scholar": search_google_scholar,
        "search_youtube": search_youtube,
    }
)

# Create research manager agent
research_manager = GPTAssistantAgent(
    name="research_manager",
    llm_config = {
        "config_list": config_list,
        "assistant_id": "asst_EBUqG2qgddkIMzxR2Nh6Z25M"
    }
)


# Create director agent
director = GPTAssistantAgent(
    name = "director",
    llm_config = {
        "config_list": config_list,
        "assistant_id": "asst_nxvBahLgeGUqzROL7GQaV6Wi",
    }
)

director.register_function(
    function_map={
        "get_airtable_records": get_airtable_records,
        "update_single_airtable_record": update_single_airtable_record
    }
)


# Create group chat
groupchat = autogen.GroupChat(agents=[user_proxy, researcher, research_manager, director], messages=[], max_round=15)
group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})


# # ------------------ start conversation ------------------ #
# message = """
# Research the dieta recommendations noya rao
# """
# user_proxy.initiate_chat(group_chat_manager, message=message)

if __name__ == "__main__":

    def main():
        message = input("Enter your message: ")
        response = user_proxy.initiate_chat(group_chat_manager, message=message)
        print("Assistant response:", response)

    main()



# if __name__ == "__main__":

#     def main():
        
#         query = input("Enter your query for youtube: ")
#         print("Searching youtube for:", query)
#         results = search_youtube(query)
#         print("Results:", results)

#     main()

