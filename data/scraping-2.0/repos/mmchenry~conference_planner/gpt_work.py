"""
Functions that prepare data for and run the GPT-4 model, which rates the applicability of keywords to abstracts.
"""

import openai
import os
import pandas as pd
import re

# Formulate the prompt text for GPT-4
def give_prompt(keywords, abstract_text):

    prompt_text = f"Give a 1 sentence summary of the following biological abstract, excluding words like 'this abstract says . . .'. State this summary after 'summary - ', followed by carriage return. Also, rate how well the text matches the following subjects: {keywords}. For each subject, rate its match to the text on a scale of 0 to 1. Report your ratings with 'subjects - ' carriage return, then state each subject and give each rating after a colon and separate each rating by a comma. Here's the text:\n {abstract_text}"

    return prompt_text


def analyze_abstracts(data_root, api_key_path, max_attempts=5):
    """
    Analyzes abstracts by calling the GPT-4 model and updates each abstract with a summary and keyword ratings.

    Parameters:
    - interdata_path (str): The directory path where intermediate data files are stored. This directory should contain 
                            CSV files in the subdirectories 'contributed_ratings' and 'poster_ratings'.
    - api_key (str): The API key used to authenticate requests to the OpenAI API.
    - max_attempts (int, optional): The maximum number of attempts to retry calling the GPT-4 model for each abstract.

    Process:
    - The function prompts the user for confirmation before proceeding, as it incurs charges for using ChatGPT.
    - It iterates over the specified subdirectories, processing CSV files containing abstracts.
    - For each abstract, it checks if all keyword columns are empty (indicating that they haven't been analyzed).
    - If keyword columns are empty, it constructs a prompt and calls `interact_with_gpt4` to analyze the abstract.
    - The function retries the GPT-4 call up to the specified number of max_attempts if the initial attempts fail.
    - After successful analysis or reaching the maximum number of attempts, it updates the CSV file with new data.

    Exceptions:
    - If the maximum number of attempts is reached without successful analysis, it raises a RuntimeError.

    Returns:
    - None: The function saves updates directly to the CSV files and does not return a value.

    Note:
    - This function should be used with caution due to potential costs associated with the OpenAI API usage.
    - The function updates the CSV files in-place and provides feedback on the process in the console.
    """

    # Check that this function should be run
    response = input("Caution: This function will incur charges to use ChatGPT. Proceed? (y/n): ")
    if response.lower() != 'y':
        print("Function canceled.")
        return
    
    # read the text at path_to_API_key to define open_api_key
    with open(api_key_path, 'r') as file:
        api_key = file.read()

    # Filenames that hold the abstract and keyword csv files
    ratings_files = ['talks_ratings', 'posters_ratings']

    # Get listing of all division subdirectories
    div_path = os.path.join(data_root, 'division_files')
    divisions = [dir for dir in os.listdir(div_path) if os.path.isdir(os.path.join(div_path, dir))]

    # Loop through talks and posters
    for csv_file in ratings_files:

        # Loop through each csv file
        for division in divisions:

            # Read the abstract csv file
            curr_dir = os.path.join(data_root, 'division_files', division)
            csv_path = os.path.join(curr_dir, csv_file + '.csv')
            df = pd.read_csv(csv_path)

            print(f"Processing {csv_file} for division {division} ...")

            # Get a list of all keyword columns
            keywords = [col for col in df.columns if col not in ['id', 'clean_title', 'clean_abstract', 'summary']]

            # Loop through each row of df
            for i, row in df.iterrows():

                # Execute only if all keywords columns are empty
                if row[keywords].isnull().all():
                    abstract_text = row['clean_abstract']
                    prompt_text = give_prompt(keywords, abstract_text)

                    # Initialize attempt counter
                    attempt = 0

                    # Try to obtain the response up to max_attempts times
                    while attempt < max_attempts:
                        try:
                            updated_row = interact_with_gpt4(prompt_text, api_key, row.copy())
                            if updated_row[keywords].isnull().all():
                                print(f"Attempt {attempt+1} failed, retrying...")
                                attempt += 1
                            else:
                                # Update the DataFrame at the current row if successful
                                df.iloc[i] = updated_row
                                break
                        except Exception as e:
                            print(f"Attempt {attempt+1} encountered an error: {e}")
                            attempt += 1

                    # Check if the maximum number of attempts was reached
                    if attempt == max_attempts:
                        raise RuntimeError(f"  Maximum attempts reached for abstract {i+1}. Unable to retrieve ratings.")
                    
                    # Save the DataFrame at the end of each abstract
                    df.to_csv(csv_path, index=False)
                
                # Report completion of abstract
                print(f"   abstract {i+1} completed. {len(df) - (i+1)} remaining.")

        # Report completion of division
        print(f"Completed {csv_file} for division {division}.")

    # Report completion of all divisions
    print(" ")
    print("Keywords ratings for all divisions completed.")



def interact_with_gpt4(prompt_text, api_key, row):
    """
    Interacts with GPT-4 using a given prompt and updates a DataFrame row with the extracted summary and keyword ratings.

    Parameters:
    - prompt_text (str): The prompt text to be sent to GPT-4 for generating the response.
    - api_key (str): The API key for authenticating requests with the OpenAI API.
    - row (pd.Series): A Pandas Series object representing a row of a DataFrame. This row should contain columns 
                       corresponding to various keywords and a 'summary' column.

    Process:
    - Sets the OpenAI API key and sends a request to the GPT-4 model with the provided prompt.
    - Extracts the response text from GPT-4.
    - Searches for a 'Summary - ' section followed by a 'Subjects - ' section in the response (case insensitive).
    - Extracts the summary text and updates the 'summary' column in the row.
    - Extracts the values for each keyword mentioned in the 'Subjects - ' section.
    - Updates the row with the extracted values for each corresponding keyword.
    - Keywords are normalized to match the DataFrame column names (lowercase, spaces replaced with underscores).

    Exceptions:
    - Raises a ValueError if the expected 'Summary - ' or 'Subjects - ' format is not found in the response.

    Returns:
    - row (pd.Series): The updated Pandas Series with the summary and keyword values extracted from the GPT-4 response.

    Note:
    - This function requires a valid OpenAI API key and depends on the structure of the GPT-4 response.
    - It's tailored to work with responses that contain specific sections ('Summary' and 'Subjects').
    """

    # Set OpenAI API key
    openai.api_key = api_key

    # Attempt the GPT-4 query
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an intelligent assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=700
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    # Extract response text 
    response_text = response['choices'][0]['message']['content']

    # Raise error, if the word 'Summary - ' is not found in the response (case insensitive)
    if not re.search(r'Summary\s*-\s*', response_text, re.I):
        raise ValueError("The expected format 'Summary - ... ' was not found in the response.")
    
    # Raise error, if the word 'Subjects - ' is not found in the response (case insensitive)
    if not re.search(r'Subjects\s*-\s*', response_text, re.I):
        raise ValueError("The expected format '... Subjects - ...' was not found in the response.")

    # Extract summary text
    match_summary = re.search(r'Summary\s*-\s*(.*?)\s*\n*[Ss]ubjects\s*-', response_text, re.S | re.I)
    if match_summary:
        summary_text = match_summary.group(1)
        row['summary'] = summary_text
    else:
        raise ValueError("The expected format 'Summary - ... Subjects - ...' was not found in the response.")

    # Extract keywords and their values, ensuring the first keyword is captured
    subjects_start = re.search(r'[Ss]ubjects\s*-', response_text)
    if subjects_start:
        subjects_text = response_text[subjects_start.end():]
        match_keywords = re.findall(r'(\w[\w\s\-]*):\s*(\d+(?:\.\d+)?)', subjects_text, re.I)
        for keyword, value in match_keywords:
            formatted_keyword = keyword.strip().lower().replace('-', ' ')
            if formatted_keyword in row:
                row[formatted_keyword] = float(value)

    return row




