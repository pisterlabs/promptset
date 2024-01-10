import pandas as pd
import concurrent.futures
import os
import openai
import dotenv
from tqdm import tqdm
from module.agent import get_meme_binary_classification
from module.prompts import *

# Load the .env file
dotenv.load_dotenv(".env")
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Read the data
data = pd.read_csv("data/train.csv")
error_df = pd.DataFrame(columns=data.columns)
checkpoint_dir = "checkpoints"
checkpoint_interval = 100  # Save after every 100 entries

def process_entry(index, entry):
    try:
        clean_data = {
            "Text": entry['text'],
            "Image Caption": entry['image_caption'],
            "Surface Message": entry['surface_message'],
            "Background Knowledge": entry['background_knowledge'],
        }
        result = get_meme_binary_classification(clean_data)

        if result is None:
            return index, None

        processed_data = {
            'prompt': result['prompt'],
            'result_text': result['text'],
            'predicted_answer': result['answer']
        }

        return index, processed_data
    except Exception as e:
        print(f"An error occurred while processing entry {index}: {e}")
        return index, None

def write_to_csv(processed_data, error_indices):
    # Write processed data and error indices to CSV files
    global data, error_df
    for index, result in processed_data:
        if result:
            for key, value in result.items():
                data.at[index, key] = value
        else:
            error_df = error_df.append(data.iloc[index])

    data.to_csv(f"{checkpoint_dir}/data_checkpoint.csv", index=False)
    error_df.to_csv(f"{checkpoint_dir}/error_checkpoint.csv", index=False)

# Parallel processing
processed_data = []
error_indices = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit all the tasks and create a list of futures
    futures = {executor.submit(process_entry, i, row): i for i, row in data.iterrows()}

    # Progress bar
    with tqdm(total=len(data), desc="Processing entries") as progress:
        for future in concurrent.futures.as_completed(futures):
            index, result = future.result()
            processed_data.append((index, result))

            if len(processed_data) >= checkpoint_interval:
                write_to_csv(processed_data, error_indices)
                processed_data = []
                error_indices = []

            progress.update(1)

# Final write to CSV outside the loop
write_to_csv(processed_data, error_indices)