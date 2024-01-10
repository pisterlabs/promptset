import openai
import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from tqdm import tqdm

# Load API key from environment variables
load_dotenv(find_dotenv())
openai.api_key = ''

def get_completion(prompt, model="gpt-4"):
    """
    Generate completion using the OpenAI Chat API.

    Parameters:
    - prompt: The user's input prompt.
    - model: The OpenAI model to use (default is "gpt-4").

    Returns:
    - The generated completion.
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # Degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

# Read evaluation data from CSV file
# Changed absolute path to relative path
eval_df = pd.read_csv("demo_data/eval_files/eval_gpt.csv")

# Iterate through rows and generate completions
for index, row in tqdm(eval_df.iterrows(), total=eval_df.shape[0]):
    prompt_top2_citance_embed_bm25 = f"""Generate a coherent summary for the following scientific text in not more than 5 sentences:  {row["top2_citance_embed_bm25"]}"""

    print(f"Prompt: {prompt_top2_citance_embed_bm25}")
    if pd.isna(eval_df.at[index, "summary_top2_citance_embed_bm25"]):
        response1 = get_completion(prompt_top2_citance_embed_bm25)
        print(f"Response: {response1}\n")
        eval_df.at[index, "summary_top2_citance_embed_bm25"] = response1

    prompt_top5_citance_embed_bm25 = f"""Generate a coherent paraphrased text for the following scientific text:  {row["top5_citance_embed_bm25"]}"""
    print(f"Prompt: {prompt_top5_citance_embed_bm25}")
    if pd.isna(eval_df.at[index, "summary_top5_citance_embed_bm25"]):
        response2 = get_completion(prompt_top5_citance_embed_bm25)
        print(f"Response: {response2}\n")
        eval_df.at[index, "summary_top5_citance_embed_bm25"] = response2

    prompt_top2_citance_single_sbert = f"""Generate a coherent summary for the following scientific text in not more than 5 sentences:  {row["top2_citance_single_sbert"]}"""
    print(f"Prompt: {prompt_top2_citance_single_sbert}")
    if pd.isna(eval_df.at[index, "summary_top2_citance_single_sbert"]):
        response3 = get_completion(prompt_top2_citance_single_sbert)
        print(f"Response: {response3}\n")
        eval_df.at[index, "summary_top2_citance_single_sbert"] = response3

    prompt_top5_citance_single_sbert = f"""Generate a coherent paraphrased text for the following scientific text:  {row["top5_citance_single_sbert"]}"""
    print(f"Prompt: {prompt_top5_citance_single_sbert}")
    if pd.isna(eval_df.at[index, "summary_top5_citance_single_sbert"]):
        response4 = get_completion(prompt_top5_citance_single_sbert)
        print(f"Response: {response4}\n")
        eval_df.at[index, "summary_top5_citance_single_sbert"] = response4

    # Save the updated DataFrame to the CSV file
    # Changed absolute path to relative path
    eval_df.to_csv("demo_data/eval_files/eval_gpt.csv", index=False)
