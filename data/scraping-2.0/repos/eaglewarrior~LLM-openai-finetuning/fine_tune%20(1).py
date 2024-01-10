import json


import pandas as pd
import tiktoken
from openai import OpenAI


def create_train_val_test_split(medical_report):
    """Create train, validation and test splits from the data

    Args:
        medical_report (df): Dataframe containing the data

    Returns:
        tuple: Train, validation and test dataframes
    """
    grouped_data = medical_reports.groupby("medical_specialty").sample(110, random_state=42) # Sample 110 items from each class

    val_test_data = grouped_data.groupby("medical_specialty").sample(10, random_state=42)  # sample 10 items from the above data
    val = val_test_data.groupby("medical_specialty").head(5) # Take the first 5 of each class
    test = val_test_data.groupby("medical_specialty").tail(5) # Take the last 5 of each class

    train = grouped_data[~grouped_data.index.isin(val_test_data.index)] # Take the remaining ones for training

    return train, val, test


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string.
    (https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)"""
    encoding = tiktoken.get_encoding("cl100k_base")  # encoding for currently all models
    num_tokens = len(encoding.encode(string))
    return num_tokens

def calculate_price(train_df, price_model=0.0080):
    """Calculate the price for fine-tuning the model

    Args:
        train_df (dataframe): Dataframe containing the training data
        price_model (float, optional): Price per 1k tokens. Defaults to 0.0080.
    """
    report_lengths = train_df['report'].apply(lambda x: num_tokens_from_string(x))
    avg_report_length = report_lengths.mean()
    min_report_length = report_lengths.min()
    max_report_length = report_lengths.max()
    report_length_sum = report_lengths.sum()

    print(f"Average report length: {avg_report_length:.2f} tokens")
    print(f"Minimum report length: {min_report_length} tokens")
    print(f"Maximum report length: {max_report_length} tokens")
    print(f"# The training dataset consists of: {report_length_sum} tokens")

    price_per_epoch = (report_length_sum / 1000) * price_model 
    print(f"Fine-tuning costs ~ ${price_per_epoch:.2f} per epoch") 

def ask_to_continue():
    """Ask the user if they want to continue"""
    answer = input("Do you want to continue? [y/n]: ")
    if answer == 'y':
        return True
    else:
        return False
    
def df_to_format(df, system_prompt):
    """Convert the dataframe to the required format

    Args:
        df (dataframe): Dataframe containing the data
        system_prompt (string): System prompt for the chat model

    Returns:
        list: List of dictionaries containing the formatted data
    """
    formatted_data = []
    
    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        entry = {"messages": [{"role": "system", "content": system_prompt},
                              {"role": "user", "content": row["report"]},
                              {"role": "assistant", "content": row["medical_specialty"]}]}

        formatted_data.append(entry)

    return formatted_data


def to_jsonl(data, filename):
    """Write the data to a jsonl file

    Args:
        data (list): List of dictionaries containing the data
        filename (string): Name of the file to write to
    """
    with open(filename, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry))
            f.write("\n")

def check_num_tokens(prompt):
    """Check if the number of tokens in the prompt exceeds the limit

    Args:
        prompt (chatPrompt): Prompt to check

    Returns:
        bool: True if the number of tokens is less than 4000, False otherwise
    """
    prompt_text = " ".join([content["content"] for content in prompt["messages"]])
    tokens = num_tokens_from_string(prompt_text)
    if tokens > 4000: # according to https://platform.openai.com/docs/guides/fine-tuning/token-limits
        print(f"Prompt {prompt} exceeds token limit!")
        return False
    return True

def check_prompt(prompt):
    """Check if the prompt is valid

    Args:
        prompt (chatPrompt): Prompt to check

    Returns:
        bool: True if the prompt is valid, False otherwise
    """
    if len(prompt["messages"][1]["content"]):
        if len(prompt["messages"][2]["content"]):
            return True
    print(f"Prompt {prompt} is missing data!")
    return False

def validate_data(path_to_jsonl):
    with open(path_to_jsonl, 'r') as f:
        dataset = [json.loads(line) for line in f]
    for element in dataset:
        assert check_num_tokens(element) and check_prompt(element)
        



if __name__ == '__main__':

    # Load the data
    medical_reports = pd.read_csv('reports.csv')  # Change accordingly to your data
    
    # Dropping rows where 'report' is missing
    medical_reports.dropna(subset=['report'], inplace=True)
    medical_reports.info()

    # Create train, validation and test splits
    train, val, test = create_train_val_test_split(medical_reports)

    # Calculate the price for fine-tuning
    calculate_price(train)

    # Ask the user if they want to continue
    if not ask_to_continue():
        print("Exiting...")
        exit()

    # Define the system prompt
    system_prompt = "Given the medical description report, classify it into one of these categories: " + \
                 "[Cardiovascular / Pulmonary, Gastroenterology, Neurology, Radiology, Surgery]"
    
    # Convert the dataframes to the required format
    train_formatted = df_to_format(train, system_prompt)

    # Write the data to jsonl files
    to_jsonl(train_formatted, "train_data.jsonl")

    # Perform the same steps for validation data
    val_formatted = df_to_format(val, system_prompt)
    to_jsonl(val_formatted, "val_data.jsonl")

    # Validate the data
    validate_data("train_data.jsonl")
    validate_data("val_data.jsonl")

    ################## TRAINING ##################
    client = OpenAI(api_key="")

    # Upload train files to OpenAI
    file_upload_response = client.files.create(
                            file=open("train_data.jsonl", "rb"),
                            purpose='fine-tune')
    # Upload validation files to OpenAI
    file_upload_response_val = client.files.create(
                            file=open("val_data.jsonl", "rb"),
                            purpose='fine-tune')
    # Start fine-tuning
    fine_tuning_response = client.fine_tuning.jobs.create(training_file=file_upload_response.id,
                            model="gpt-3.5-turbo",
                            hyperparameters={"n_epochs": 1},
                            validation_file = file_upload_response_val.id)
