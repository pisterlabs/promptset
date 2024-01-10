import pandas as pd
import openai
import os
import sys
import time
import concurrent.futures

# Add the path to the custom module directory
new_path = '/Users/williamfussell/Documents/Github/amasum/src/'
if new_path not in sys.path:
    sys.path.append(new_path)

from sanitization import *  # Custom sanitization module

# Function to generate a summary using the OpenAI API with a timeout mechanism
def generate_summary(review_text, timeout=45, max_length=1024):
    """
    Generate a concise summary of a given review text using OpenAI API with a timeout.

    Parameters:
    review_text (str): The review text to be summarized.
    timeout (int): The maximum time in seconds to wait for a response.
    max_length (int): Maximum length of review text to process.

    Returns:
    str: A summary of the review text or None if it times out.
    """
    # Truncate the review text if it's too long
    if len(review_text) > max_length:
        review_text = review_text[:max_length] + '...'

    prompt = f"Summarize the following review in no more than 120 words:\n\n{review_text}\n\nProvide three bullet points highlighting the most discussed topics and their associated issues. Keep the summary concise and focused on the key points."
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            openai.ChatCompletion.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            api_key=api_key
        )
        try:
            response = future.result(timeout=timeout)
            return response['choices'][0]['message']['content'].strip()
        except concurrent.futures.TimeoutError:
            print("Request timed out.")
            return None

# Set API key for OpenAI
api_key = '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'

# Read in the dataframes
df_neg = pd.read_json('/Users/williamfussell/Downloads/df_neg_sample.json')
df_pos = pd.read_json('/Users/williamfussell/Downloads/df_pos_sample.json')
print("Dataframes loaded successfully.")

# Add a 'summary' column to the dataframes if it doesn't exist
for df in [df_neg, df_pos]:
    if 'summary' not in df.columns:
        df['summary'] = None

# Function to process reviews and generate summaries
def process_reviews(df, json_path):
    print(f"Processing reviews...")
    processed_count = 0
    for index, row in df.iterrows():
        if pd.isna(row['summary']):
            try:
                summary = generate_summary(row['reviewText'])
                if summary:
                    df.at[index, 'summary'] = summary  # Update the dataframe
                    processed_count += 1
                else:
                    print(f"Skipping review at index {index} due to timeout.")

                if processed_count % 2 == 0:  # After every 2 requests
                    df.to_json(json_path)
                    print(f"{processed_count} reviews processed and saved. Waiting 30 seconds before next request.")
                    time.sleep(20)  # Wait for 20 seconds

            except Exception as e:
                print(f"An error occurred: {e}")
    # Save the final DataFrame
    df.to_json(json_path)
    print(f"Processing complete. A total of {processed_count} reviews were processed.")

# Process negative reviews
process_reviews(df_neg, '/Users/williamfussell/Downloads/negative_review_summaries_final.json')

# Process positive reviews
process_reviews(df_pos, '/Users/williamfussell/Downloads/positive_review_summaries_final.json')




