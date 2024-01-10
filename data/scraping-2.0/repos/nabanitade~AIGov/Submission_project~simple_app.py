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
import evaluate_datasets 
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
    st.title("NYC Open Data App")

    page = st.sidebar.selectbox("Choose a page:", ["Data Fetcher", "Data Viewer", "File Ranking per Dataset"])


    if page == "Data Fetcher":
        st.header("Data Fetcher")
        
        query = st.text_input("Enter your query:", value="311")  # Default value is 311
        fetch_button = st.button("Fetch Data")

        if fetch_button:
            url = f'https://data.cityofnewyork.us/browse?q={query}'
            folder = f'{query}_data'
            endpoints = await fetch_endpoints_from_url(url)
            save_endpoints_to_files(endpoints, folder)
    
    elif page == "Data Viewer":
        st.header("Data Viewer")

        # List the folders containing '_data' in their names
        folders = [f for f in os.listdir('.') if os.path.isdir(f) and '_data' in f]
        folder_choice = st.selectbox("Select a folder:", folders)

        if folder_choice:
            # List the datasets within the selected folder
            datasets = [file for file in os.listdir(folder_choice) if file.endswith('.json')]
            dataset_choice = st.selectbox("Select a dataset:", datasets)

            if dataset_choice:
                # Load the selected dataset
                file_path = os.path.join(folder_choice, dataset_choice)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                df = pd.DataFrame(data)
                # Display basic statistics of the dataset
                st.subheader("Summary Statistics")
                st.write(df.describe())
                
    elif page == "File Ranking per Dataset":
        st.header("File Ranking per Dataset")

        # List the folders containing '_data' in their names
        folders = [f for f in os.listdir('.') if os.path.isdir(f) and '_data' in f]
        folder_choice = st.selectbox("Select a folder:", folders)

        if folder_choice:
            # Call the rank_files_based_on_scores function for the selected folder
            
            ranked_files = evaluate_datasets.rank_files_based_on_scores(folder_choice)
            
            # Display the ranked files and their scores
            st.write("Ranked Files:")
            # for rank, file_name in enumerate(ranked_files, 1):
            #     st.write(f"{rank}. {file_name}")
            for i in ranked_files:
                st.write(i)
            # for rank, file_name in enumerate(ranked_files, 1):
            #     st.write(f"{rank}. {file_name} - Completeness: {scores[file_name]['completeness']:.2f}, Variability Avg: {scores[file_name]['variability_avg']:.2f}")
    
# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
