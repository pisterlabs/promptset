#chat_completion_with_retries and get_embedding_with_retries based on https://github.com/aypan17/machiavelli/blob/723d76c9db388e3590703cbd7d5443f452969464/machiavelli/openai_helpers.py
import os
import time
import openai
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from typing import Dict, List, Mapping, Optional
import re
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Set OpenAI API key
ENV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=ENV_PATH)
openai.api_key = os.getenv("OPENAI_API_KEY")

def chat_completion_with_retries(model: str, messages: List, max_retries: int = 5, retry_interval_sec: int = 20, api_keys=None, validation_fn=None, **kwargs) -> Mapping:
    for n_attempts_remaining in range(max_retries, 0, -1):
        try:
            res = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                **kwargs)
            
            if validation_fn is not None:
                if not validation_fn(res["choices"][0]["message"]["content"]):
                    print(f"VALIDATION FAILED!\nTrying again.")
                    continue
            return res
        except (
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.Timeout,
            openai.error.TryAgain,
            openai.error.OpenAIError,
            ) as e:
            print(e)
            print(f"Hit openai.error exception. Waiting {retry_interval_sec} seconds for retry... ({n_attempts_remaining - 1} attempts remaining)", flush=True)
            time.sleep(retry_interval_sec)
    return {}

def save_labels(path, rows, columns=None, mode='a', header=False):
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, mode=mode, header=header, index=False)
    
def get_last_processed_row(path, default_header=None):
    try:
        df = pd.read_csv(path)
        return len(df)
    except FileNotFoundError:
        if default_header:
            with open(path, 'w') as f:
                f.write(default_header + "\n")
        return 0

'''
Pattern:
A number, followed by a dot, then a space.
Any sequence of characters (non-greedy).
Ending with a colon.
'''

def extract_pattern(text):
    expressions = re.search(r'(\d+)\.\s([^:]+):', text) # Finds first item
    if not expressions:
        # get first 10 space-separated items in the text as a single string
        return ' '.join(text.split(" ")[:10])
    return expressions.group(2)

def get_embedding_with_retries(text, model: str ="text-embedding-ada-002", max_retries: int = 5, retry_interval_sec: int = 10):
    #text = extract_pattern(text) # We are temporarily changing this function so that the patterns/first 10 items are extracted from the text
    for n_attempts_remaining in range(max_retries, 0, -1):
        try:
            res = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
            return res
        except (
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.Timeout,
            openai.error.TryAgain,
            openai.error.OpenAIError,
            ) as e:
            print(e)
            print(f"Hit openai.error exception. Waiting {retry_interval_sec} seconds for retry... ({n_attempts_remaining - 1} attempts remaining)", flush=True)
            time.sleep(retry_interval_sec)
    return []

def process_csv_and_plot(target_word, input_filepath, output_csv_filepath, output_image_filepath):
    # Load the CSV file
    df = pd.read_csv(input_filepath)
    
    # Create a new column 'labels' based on the presence of the target_word in the 'completion' column
    df['labels'] = df['completion'].str.contains(target_word, case=False).astype(int)
    
    # Save the modified dataframe to the provided output filepath
    df.to_csv(output_csv_filepath, index=False)
    
    # Count the number of completions where target_word appears for different unique 'context_key' values
    context_counts = df.groupby('context_key')['labels'].sum().sort_values()
    
    # Save the bar chart as a PNG image file
    plt.figure(figsize=(15, 10))
    context_counts.plot(kind='barh')
    plt.xlabel('No. of completions where {target_word} appears')
    plt.ylabel('Context')
    plt.title(f'Occurrences of {target_word} by context')
    plt.tight_layout()
    plt.savefig(output_image_filepath)
    plt.close()

def bootstrap_and_plot(input_dict, plot_title, x_label, y_label, output_path):    
    # Step 1: Create a list with each key repeated according to its value
    data_list = [k for k, v in input_dict.items() for _ in range(v)]
    
    # Step 2: Bootstrap sampling and calculating the means
    n_bootstrap = 1000
    sample_size = sum(input_dict.values())
    bootstrap_means = {k: [] for k in input_dict.keys()}
    
    for i in range(n_bootstrap):
        bootstrap_sample = resample(data_list, replace=True, n_samples=sample_size, random_state = i)
        bootstrap_counts = {k: bootstrap_sample.count(k) for k in input_dict.keys()}
        for k, v in bootstrap_counts.items():
            bootstrap_means[k].append(v)
    
    # Step 3: Calculate the estimated mean and its 95% confidence interval
    confidence_intervals = {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in bootstrap_means.items()}
    
    # Step 4: Get the original counts and keys, and sort them by counts in descending order
    original_counts = list(input_dict.values())
    keys = list(input_dict.keys())
    sorted_indices = np.argsort(original_counts)
    sorted_keys = np.array(keys)[sorted_indices]
    sorted_counts = np.array(original_counts)[sorted_indices]
    
    # Step 5: Get the sorted lower and upper bounds of the confidence intervals
    lower_bounds = [confidence_intervals[key][0] for key in sorted_keys]
    upper_bounds = [confidence_intervals[key][1] for key in sorted_keys]

    # Step 6: Plotting the sorted bar chart with horizontal bars
    plt.figure(figsize=(10, 12))
    error_bars = [[sorted_counts[i] - lower_bounds[i], upper_bounds[i] - sorted_counts[i]] for i in range(len(sorted_keys))]
    plt.barh(sorted_keys, sorted_counts, xerr=np.array(error_bars).T, color='skyblue', capsize=3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.grid(axis='x')
    
    # Adjusting the layout to prevent clipping of labels
    plt.tight_layout()

    # Save the plot to the specified output path
    plt.savefig(output_path)
    #plt.show()