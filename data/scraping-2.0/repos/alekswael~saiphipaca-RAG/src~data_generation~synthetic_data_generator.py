import openai
import pandas as pd
import os

# Set your OpenAI API key
openai.api_key = "YOUR-API-KEY"

def gpt4_conversation(conversation_log):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=conversation_log,
        temperature=0
    )
    conversation_log.append({
        'role': response.choices[0].message.role,
        'content': response.choices[0].message.content.strip()
    })
    response = conversation_log[-1]['content']
    return response

# Defining a path for the file that will host all processed pages until the full dataset of research papers has been processed
# The role of this file is that in case the Open API timesout, the script will begin from the page/document it left off

if os.path.exists(partial_file_path):
    # Load the partial dataset
    dataset = pd.read_csv(partial_file_path)
    print("Loaded the partial dataset from synthetic_dataset_partial.csv.")
else:
    # Load your original dataset
    dataset = pd.read_csv("synthetic_data_raw.csv")  # Replace "synthetic_data_raw.csv" with your actual dataset file

    # Calculate the 25th percentile of text lengths
    quantile_25 = dataset['text'].apply(len).quantile(0.25)

    # Create a mask for rows that are shorter than 25th quantile to be removed
    mask = dataset['text'].apply(len) < quantile_25

    # Remove rows based on the mask
    dataset = dataset[~mask]

    # Add an index to the DataFrame
    dataset = dataset.reset_index(drop=True)

    # Add a new column to mark synthetic data
    dataset['text_synthetic'] = ''

# Define the prompt for generating questions
question_prompt = "Generate a question based on the following text:\n\n"

# Define the prompt for answering questions
answer_prompt = "Answer the following question based on the given text:\n\n"

# Iterate through each row in the dataset
for index, row in dataset.iterrows():

    # Check if 'text_synthetic' column is already populated
    if not pd.isnull(row["text_synthetic"]) and row["text_synthetic"] != '':
        print(f"Skipping row {index} as it is already processed.")
        continue

    print(f"Processing row {index} out of {len(dataset)}")

    text = row["text"]

    conversation_log = [{'role': 'user', 'content': text + question_prompt}]
    question = gpt4_conversation(conversation_log)

    conversation_log = [{'role': 'user', 'content': text + answer_prompt + question}]
    answer = gpt4_conversation(conversation_log)

    # Update the 'text_synthetic' column
    dataset.at[index, "text_synthetic"] = f"Text: {text}\nHuman: {question}\nAssistant: {answer}\n\n"

    # Save the modified dataset with the new column after each row
    dataset.to_csv("synthetic_dataset_partial.csv", index=False)

# Save the final modified dataset with the new column
dataset.to_csv("synthetic_data.csv", index=False)
