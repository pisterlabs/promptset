import streamlit as st
from utils import populate_user_info, load_and_validate_env
import requests
import json
import pandas as pd
from pandas.errors import ParserError
from newspaper import Article
import openai
from retry import retry
from io import StringIO
from json.decoder import JSONDecodeError
from uuid import uuid4
from datetime import datetime
from openai.error import InvalidRequestError

from azure.cosmos import CosmosClient, PartitionKey

# Page Config
st.set_page_config(
    page_title = "GeoWise", 
    page_icon = "🌍", 
    initial_sidebar_state = "auto",
    layout = "wide"
)
st.header("🌍 GeoWise - Extraction")
st.sidebar.info("ChatGPT augmented geocoding for unstrucutred text with Azure Maps")
user_details = populate_user_info()
project_name = st.sidebar.text_input("Project to save Data under", value="Default")

# Environment Variables

ENV = {
    "AZURE_OPENAI_API_ENDPOINT": "",
    "AZURE_OPENAI_API_VERSION": "",
    "AZURE_OPENAI_SERVICE_KEY": "",
    "AZURE_OPENAI_CHATGPT_DEPLOYMENT": "",
    "AZURE_OPENAI_GPT4_DEPLOYMENT": "",
    "AZURE_MAPS_KEY": "",
    "COSMOSDB_URL": "",
    "COSMOSDB_KEY": "",
    "COSMOSDB_DATABASE_NAME": "",
    "COSMOSDB_CONTAINER_NAME": ""
}

ENV = load_and_validate_env(ENV)

# OpenAI API Setup

openai.api_key = ENV["AZURE_OPENAI_SERVICE_KEY"]
openai.api_type = "azure"
openai.api_base = ENV["AZURE_OPENAI_API_ENDPOINT"]
openai.api_version = ENV["AZURE_OPENAI_API_VERSION"]

# Model Selector
model_dict = {
    ENV["AZURE_OPENAI_GPT4_DEPLOYMENT"]: "GPT-4 32K",
    ENV["AZURE_OPENAI_CHATGPT_DEPLOYMENT"]: "ChatGPT"
}
model_selector = st.sidebar.radio("Select a model", list(model_dict.keys()), format_func=lambda x: model_dict[x], index=1)

# CosmosDB Setup

cosmos_client = CosmosClient(url=ENV["COSMOSDB_URL"], credential=ENV["COSMOSDB_KEY"])
cos_db_client = cosmos_client.get_database_client(ENV["COSMOSDB_DATABASE_NAME"])
cos_container_client = cos_db_client.get_container_client(ENV["COSMOSDB_CONTAINER_NAME"])

# Functions

def remove_nan_values(d):
    """Recursively remove all nan values from a dictionary"""
    for k, v in d.items():
        if isinstance(v, dict):
            remove_nan_values(v)
        if isinstance(v, float):
            if pd.isna(v):
                d[k] = None
    return d

@retry(tries=5, delay=5)
def call_completion(messages):
    """Call the OpenAI API to get a completion with intelligent retrying"""
    try:
        
        response = openai.ChatCompletion.create(
            engine=model_selector,
            messages=messages,
            request_timeout=60,
        )
        if response.choices[0].finish_reason != "stop":
            raise Exception("Completion did not finish. Reason: " + response.choices[0].finish_reason)
        return response
    except Exception as err:
        print(err)
        raise err

@st.cache_data
def search_azure_maps(search_term, country_code=None):
    """Search for a location using Azure Maps"""

    search_url = f"https://atlas.microsoft.com/search/fuzzy/json?api-version=1.0&query={search_term}&subscription-key={ENV['AZURE_MAPS_KEY']}"

    # Limit to a country if specified
    if country_code is not None:
        search_url += f"&countrySet={country_code}"
    else:
        pass

    resp = requests.get(
        search_url,
        timeout=30
    )

    return json.loads(resp.text)["results"]

def extract_location_events_from_text(input_text):
    """Get a list of locations and events from a text input using LLM"""

    # Get GPT4 to give us a list of locations mentioned

    location_extraction_prompt = '''From the following article, please extract a CSV table with ',' \
        as the delimeter (like a CSV Structure) between columns and with each row being of the \
        locations mentioned in the article. The output will be used to search a geocoding API \
        where we need to submit optimal search terms (don't include long sentences). \
        The table should have only these 3 columns: location_name, event_category_at_location, event_description .\
        event_category_at_location can have one of the following values: \
        DISASTER, NATURAL_DISASTER, CRIME_EVENT, PROTEST_EVENT, POLITICAL_EVENT, REFUGEES, WARCRIME, \
        ANNOUNCEMENT_OR_SPEECH, MILITARY_EVENT, BUSINESS_EVENT, ECONOMIC_EVENT, DIPLOMATIC_EVENT, TERRORIST_EVENT, DEATH, HISTORICAL_EVENT and OTHER. \
        location_name should only be geographic places or organisations with well known geopolitical locations (e.g. "White House", "Kremlin").\
        location_name should never be a persons name, use your knowledge and article context to ignore people for the location_name field.\
        Do not respond with anything other than the  table. Include the column headers. \
        Please be as granular as possible with the locations but don't include redundant references \
        If there isn't any location mentions still return the table headers with no rows. \
        Be sure to wrap any text containing commas with quotes to maintain valid table. \
        \n\nExample (Don't include these rows in your output, but do use the column headers for your first row):\
        \n\nlocation_name,event_category_at_location,event_description\
        \n"East London, South Africa",NATURAL_DISASTER,"A wildfire is affecting East London, South Africa"\
        \n"King Phalo Airport",DIPLOMATIC_EVENT,"The UN is sending aid via King Phalo Airport"\
        \n"Corner Cheltenham Road,Selborne, East London, 5217, South Africa",ANNOUNCEMENT,\
        "The local government issued a statement from Corner Cheltenham Road"\
        \n\n Article to extract locations from:\nArticle:\n''' + input_text.replace(",", " ")

    messages = [{"role":"system","content":location_extraction_prompt}]
    response = call_completion(messages)

    if response.choices[0].finish_reason != "stop":
        raise Exception("Completion did not finish. Reason: " + response.choices[0].finish_reason)
    
    try:
        return response.choices[0].message.content
    except AttributeError as err:
        print(err)
        print(response)
        raise err

def get_location_match_candidates(query):
    """Search for a location and return a list of candidate locations"""

    search_results = search_azure_maps(query)
    if len(search_results) > 0:
        df = pd.json_normalize(search_results)

        place_dict_list = df.to_dict(orient="records")
        place_dict_list = [json.dumps(x) for x in place_dict_list]

        return place_dict_list, df
    else:
        return None, None

@st.cache_data
def download_article(article_url):
    """Download an article from a URL and return the text"""
    article = Article(article_url)
    article.download()
    article.parse()
    return article.text


def get_location_df(input_text, max_attempts=5):
    """Get a dataframe of locations from a text input extracted by LLM"""
    assert max_attempts >= 1
    current_attempt = 0

    while current_attempt < max_attempts:
        try:

            text_locations = extract_location_events_from_text(input_text)
            df = pd.read_csv(StringIO(text_locations), delimiter=",")

            if df.columns.tolist() != ["location_name", "event_category_at_location", "event_description"]:
                st.write(df)
                # Move current header to row and add new header
                new_df = pd.DataFrame([df.columns.tolist()] + df.values.tolist(), columns=["location_name", "event_category_at_location", "event_description"])
                df = new_df
                st.write(df)
            return df
        except ParserError as err:
            print(err)
            current_attempt += 1
            continue
        except InvalidRequestError as err:
            st.error("InvalidRequestError: " + str(err))
            return None
    
    st.error("Unable to parse locations from text after " + str(max_attempts))
    return None


def llm_match_location_candidates(location_name, location_candidates, article_body, max_attempts=6):
    """Get the LLM to select the most relevant location from a list of candidates"""

    assert max_attempts >= 1
    location_result_selection_prompt = '''From the list of locations below, please return the ID \
        of the most relevant location for your the search term given the context of the below article. \
        Return that ID in a JSON object with "id_choice" as a field and a "reasoning" field. If you don't \
        think there is a good match respond give an id_choice of -1 and give a reason why you think \
        there isn't a match in 'reasoning'. \
        Also give a -1 if the location proposed likely isn't an actual location (for example "John Smith" is likely invalid as a location)\
        Try to use the context of the article to determine if the location is valid. \
        Also look to find the most releant poi.category that matches the location. \
        Only return valid JSON with double quote wrapped strings. \
        Don't refer to the ID number in the reasoning, just why that location is a good match. \
        \n Example: {"id_choice": 0, "reasoning": "This is the best match because it\\'s ..."} \
        \n\nArticle Context:''' + article_body +"\n\nLocation Search Term: "+ location_name +"\n\n Location Candidates:\n"
    
    for i, location in enumerate(location_candidates):
        location_result_selection_prompt += f"\n{i}: {location}\n"

    messages = [{"role":"system","content": location_result_selection_prompt}]

    current_attempt = 0
    
    while current_attempt < max_attempts:
        try:
            raw_response = call_completion(messages)
            parsed_response = json.loads(raw_response.choices[0].message.content)
            return raw_response, parsed_response
        except JSONDecodeError as err:
            print(err)
            print(raw_response)
            current_attempt += 1
            continue


# Main Code

# Input Mode Selector

text_input_mode = st.sidebar.radio("Input Mode", ["Text Box", "File Upload", "Article URL"], index=2)

text = None

if text_input_mode == "Text Box":
    text = st.text_area("Enter text to geocode", height=200)
elif text_input_mode == "File Upload":
    text = st.file_uploader("Upload a file to geocode", type=["txt"])
elif text_input_mode == "Article URL":
    url = st.text_input("Enter an article URL to geocode")
    if url:
        with st.spinner("Downloading article..."):
            text = download_article(url)
        st.success("Article downloaded successfully")

if text is None:
    st.stop()
elif len(text) < 10:
    st.warning("Text is too short")
    st.stop()
else:
    with st.expander("Show Text"):
        st.write(text)
    run_button = st.button("Run and Upload to CosmosDB")

# Run Extraction of Locations, Geocoding and Uploading to CosmosDB

if run_button:
    # Extract Locations
    with st.spinner("📍 Extracting locations..."):
        df_text_locations = get_location_df(text)
    if df_text_locations is None:
        st.error("Unable to parse locations from text")
        st.stop()
    with st.expander("Show Extracted Locations"):
        st.write(df_text_locations)
    
    if len(df_text_locations) == 0:
        st.warning("No locations found")
        st.stop()

    # Enrich Locations

    enriched_locations = []
    progress_bar = st.progress(0)
    progress_max = len(df_text_locations)

    for i, location_dict in enumerate(df_text_locations.to_dict(orient="records")):
        location_name = str(location_dict["location_name"]).strip()

        with st.spinner("Enriching location: " + location_name):
            # Search for location and get candidate matches
            location_name_results, df_location_results = get_location_match_candidates(location_name)
            if location_name_results is None:
                st.warning(f"No results found for {location_name}")
                continue
            if len(location_name_results) == 0:
                st.warning(f"No results found for {location_name}")
                continue
        
        # Get LLM to select the most relevant location match
        with st.spinner("Selecting location for: " + location_name):
            raw_response, parsed_response = llm_match_location_candidates(location_name, location_name_results, text)

        # If no match found, skip
        if parsed_response["id_choice"] == -1:
            location_dict["geo"] = None
            location_dict["geo_reasoning"] = parsed_response["reasoning"]
            location_dict["lat"] = None
            location_dict["lon"] = None
        # If match found, add to dict
        else:
            location_dict["geo"] = df_location_results.iloc[parsed_response["id_choice"]].to_dict()
            location_dict["geo_reasoning"] = parsed_response["reasoning"]
            location_dict["lat"] = location_dict["geo"]["position.lat"]
            location_dict["lon"] = location_dict["geo"]["position.lon"]

        enriched_locations.append(location_dict)
        progress_bar.progress((i+1)/float(progress_max))
    
    progress_bar.progress(1.0)

    # Create DataFrame
    df_final = pd.DataFrame(enriched_locations)

    # Add metadata
    df_final["project_name"] = project_name
    df_final["extraction_id"] = str(uuid4())
    df_final["extraction_completion_datetime"] = datetime.utcnow().isoformat()

    # Filter out rows with no lat/lon (i.e. no match found)
    df_final_filt = df_final[df_final["lat"].notnull()]

    # Upload to CosmosDB
    documents = list(df_final_filt.to_dict(orient="records"))

    upload_progress_bar = st.progress(0)
    upload_progress_max = len(documents)
    with st.spinner("Uploading to CosmosDB"):
        for i, doc in enumerate(documents):

            # Remove all nan values within the dict
            doc = remove_nan_values(doc)
            doc = json.loads(json.dumps(doc))
            doc["id"] = str(uuid4())
            cos_container_client.create_item(body=doc)
            upload_progress_bar.progress((i+1)/float(upload_progress_max))



    

    