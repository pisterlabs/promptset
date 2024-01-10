import asyncio
import json
import os
import requests
from chatbot import Chatbot
import aiohttp
import nest_asyncio
import pandas as pd
import streamlit as st
import openai

from nyc_data_pipeline.source_extract_async import NYCPublicDataFetcher, NYCEndpointFetcher

# Apply the nest_asyncio patch
nest_asyncio.apply()

openai.api_key = os.environ.get('OPENAI_API_KEY')


async def fetch_endpoints_from_url(url):
    data_fetcher = NYCPublicDataFetcher(url)
    data_dict = await data_fetcher.run()
    data_dict = {url.split('/')[-2]: url.lower() for _, url in data_dict.items()}
    endpoint_fetcher = NYCEndpointFetcher()
    endpoints = await endpoint_fetcher.run(data_dict)
    return endpoints


def save_endpoints_to_files(endpoints, folder):
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    
    total_endpoints = len(endpoints)
    progress_bar = st.progress(0)
    progress_text = st.empty()  # Placeholder for text updates
    
    # Iterate through the endpoints dictionary
    for idx, (name, url) in enumerate(endpoints.items()):
        # Create the file path
        file_path = os.path.join(folder, f"{name}.json")
        
        # Check if data already exists
        if os.path.exists(file_path):
            progress_text.text(f'Data already exists for {name}')
        else:
            try:
                # Make the HTTP request to fetch the data
                response = requests.get(url)
                response.raise_for_status()  # Check if the request was successful
                
                # Parse the JSON data
                data = response.json()
                
                # Save the data to a file
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)
                progress_text.text(f'Successfully saved data from {name} to {file_path}')
            except requests.RequestException as e:
                progress_text.text(f'Failed to fetch data from {url}: {e}')
            except ValueError as e:
                progress_text.text(f'Failed to parse JSON data from {url}: {e}')
        
        # Update the progress bar
        progress_value = (idx + 1) / total_endpoints
        progress_bar.progress(progress_value)
    
    progress_text.text('Data fetching and saving completed.')


async def main():
    st.title("NYC Open Data Fetcher with Chatbox")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])

    chatbot = Chatbot()  # Instantiate the Chatbot

    with col1:
        query = st.text_input("Enter your query:", value="311")  # Default value is 311
        fetch_button = st.button("Fetch Data")

        if fetch_button:
            url = f'https://data.cityofnewyork.us/browse?q={query}'
            folder = f'{query}_data'
            endpoints = await fetch_endpoints_from_url(url)
            save_endpoints_to_files(endpoints, folder)
            chatbot.load_data(folder)  # Load data into the chatbot

            dataset_names = [name.replace('.json', '') for name in os.listdir(folder) if name.endswith('.json')]
            dataset_choice = st.selectbox("Select a dataset:", dataset_names)

            if dataset_choice:
                analysis_result, dashboards = chatbot.generate_response(f'{dataset_choice}.json')
                st.write("Summary Statistics:", analysis_result)
                for idx, fig in enumerate(dashboards, 1):
                    st.pyplot(fig)
    
    with col2:
        st.header("Chatbox")
        folders = [f for f in os.listdir('.') if os.path.isdir(f) and '_data' in f]
        folder_choice = st.selectbox("Select a folder:", folders)

        if folder_choice:
            insights, dashboards = chatbot.provide_insights_and_dashboards(folder_choice)
            for insight in insights:
                st.write(f"Dataset: {insight['dataset_name']}")
                st.write(f"Number of records: {insight['number_of_records']}")
                st.write(f"Number of columns: {insight['number_of_columns']}")
                st.write(f"Columns: {', '.join(insight['columns'])}")
            for dashboard in dashboards:
                st.pyplot(dashboard)

        user_input = st.text_area("Type your message here...", height=200)
        send_button = st.button("Send")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
