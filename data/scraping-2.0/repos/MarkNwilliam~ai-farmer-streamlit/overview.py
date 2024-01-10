#AIzaSyA3YVLTGuGesO27kFo1QGZq-lPNebj3ihg
from dotenv import load_dotenv
import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, ChatVertexAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms import VertexAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.chat_models import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import load_tools
#from trulens_eval import TruChain, Feedback, OpenAI, Huggingface, Tru

load_dotenv()

def get_google_maps_api_key():
    """Retrieve Google Maps API key from environment variables."""
    api_key = os.environ.get('GOOGLE_MAPS')
    if not api_key:
        raise ValueError("Google Maps API key is not set in environment variables.")
    return api_key

# Use the function to get the API key
google_maps_api_key = get_google_maps_api_key()
print("Google Maps API Key:", google_maps_api_key)


service_account_path = os.path.join(os.path.dirname(__file__), 'lablab-392213-7e18b3041d69.json')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path


#openai = OpenAI()




llm = ChatVertexAI()

tools = load_tools(["openweathermap-api", "wolfram-alpha"], llm)



# Prompt
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice farming expert chatbot having a conversation with a farmer giving advice."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)


def embed_google_map(lat, lon, api_key):
    google_maps_url = f"https://www.google.com/maps/embed/v1/view?key={api_key}&center={lat},{lon}&zoom=15"
    st.markdown(
        f'<iframe width="800" height="600" src="{google_maps_url}" frameborder="0" allowfullscreen></iframe>',
        unsafe_allow_html=True,
    )


def get_location_name(lat, lon, api_key):
    """Fetch location name from latitude and longitude using Google Geocoding API."""
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"
    response = requests.get(geocode_url)
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            response1 = results[0].get('formatted_address')
            #weather_data = weather.run()
            print(chain( response1 ))

            return results[0].get('formatted_address')

    return "Location not found"


def search_location(query, api_key):
    """Search for a location using Google Places API."""
    search_url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={query}&inputtype=textquery&fields=geometry&key={api_key}"
    response = requests.get(search_url)
    if response.status_code == 200:
        results = response.json().get('candidates', [])
        if results:
            return results[0]['geometry']['location']
    return None




def show():
    google_maps_api_key = get_google_maps_api_key()
    default_lat, default_lon = 37.7749, -122.4194  # Default coordinates (San Francisco)

    st.title("Google Map")

    # Initialize session state
    if 'lat' not in st.session_state or 'lon' not in st.session_state:
        st.session_state['lat'], st.session_state['lon'] = default_lat, default_lon

    # Search field
    query = st.text_input("Enter a location name:")
    location_updated = False
    if query:
        location = search_location(query, google_maps_api_key )
        if location:
            st.session_state['lat'] = location['lat']
            st.session_state['lon'] = location['lng']
            location_name = get_location_name(st.session_state['lat'], st.session_state['lon'], google_maps_api_key )
            st.write(f"Location: {location_name}")
            weather = OpenWeatherMapAPIWrapper()

            weather_data = weather.run("Uganda")
            print(weather_data)
            location_updated = True

    # Button to get current location
    if st.button("Get My Current Location") and not location_updated:
        with st.spinner('Fetching current location...'):
            time.sleep(2)  # Simulate delay in fetching location
            st.session_state['lat'], st.session_state['lon'] = 40.7128, -74.0060  # Example coordinates (New York City)
            location_name = get_location_name(st.session_state['lat'], st.session_state['lon'], google_maps_api_key)
            st.write(f"Location: {location_name}")
            location_updated = True

    # Embed map with updated location
    if location_updated:
        embed_google_map(st.session_state['lat'], st.session_state['lon'], google_maps_api_key)

    st.title("Contextual Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Record with TruLens

            full_response = chain.run(prompt)
            message_placeholder = st.empty()
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})


#Ai


