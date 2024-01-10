import pandas as pd
import json
import asyncio
import json
import os
import ast
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, upload_file
from parallel_processor import process_api_requests_from_file
from openai import OpenAI
import asyncio

def make_requests_file(df, filepath, model_name, fields_to_translate, target_language):
    """
    Convert a pandas DataFrame to a jsonl file with separate line for each element.

    Parameters:
    - df: pandas DataFrame to convert.
    - filepath: File path to save the jsonl file.
    - model_name: Model name to be used in the JSON object.
    - columns: List of column names in the DataFrame to be processed.
    """
    template = f"Translate to {target_language} (don't use quotes):\n"
    with open(filepath, 'w') as file:
        for row_index, row in df.iterrows():
            for col in fields_to_translate:
                content = template + str(row[col]) + "\n"
                message = [{"role": "user", "content": content}]

                # Include all other data in the metadata, except the columns specified for translation
                metadata = {key: value for key, value in row.items() if key not in fields_to_translate}
                metadata['row_index'] = row_index  # Add the row index to the metadata

                json_object = {
                    "model": model_name,
                    "messages": message,
                    "metadata": metadata
                }

                # Write the JSON string to the file
                file.write(json.dumps(json_object) + '\n')

if __name__ == "__main__":
    hf_username = "willhath"
    hf_token = "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"
    source_language = "en"
    target_languages = ["spanish"] # , "swahili", "turkish", "french", "german"
    dataset_name = "narrativeqa"

    dataset = load_dataset(dataset_name, split='test')
    # flatten the dataset if needed
    # convert the dataset to a pandas DataFrame
    df = pd.DataFrame(dataset)
    if dataset_name.endswith("clustering"):
        # only take the first 5 rows
        df = df.iloc[0:5]
        exploded = df.apply(pd.Series.explode)
        df = exploded.reset_index(drop=False)
        df = df.rename(columns={"index": "cluster_id"})
        fields_to_translate = ["sentences"]
    
    if dataset_name == "narrativeqa":
        # make a new dataframe
        # the info we need is 
        # self.queries = {self._EVAL_SPLIT: {str(i): row['question']['text'] for i, row in enumerate(data)}}
        # self.corpus = {self._EVAL_SPLIT: {str(row['document']['id']): {'text': row['document']['text']} for row in data}}
        # self.relevant_docs = {self._EVAL_SPLIT: {str(i): {row['document']['id']: 1} for i, row in enumerate(data)}}

        new_df = pd.DataFrame()
        questions = [row["question"]["text"] for i, row in df.iterrows()]
        doc_ids = [row["document"]["id"] for i, row in df.iterrows()]
        doc_texts = [row["document"]["text"] for i, row in df.iterrows()]

        new_df["question"] = questions
        new_df["doc_id"] = doc_ids
        new_df["doc_text"] = doc_texts
        df = new_df
        fields_to_translate = ["question", "doc_text"]


    for target_language in target_languages:
        # make the requests file
        requests_filepath = os.path.join("requests/", f"{target_language}-{dataset_name.split('/')[-1]}_{target_language}.jsonl")
        make_requests_file(df, requests_filepath, "gpt-3.5-turbo-1106", fields_to_translate, target_language)

        responses_filepath = os.path.join("responses/", f"{target_language}-{dataset_name.split('/')[-1]}", "response.jsonl")
        # make the directory (without the filename)
        os.makedirs(os.path.dirname(responses_filepath), exist_ok=True)
        # save the indices that have been processed in a file
        processed_indices_filepath = os.path.join("responses/", f"{target_language}-{dataset_name.split('/')[-1]}", "processed_indices.txt")


        asyncio.run(process_api_requests_from_file(requests_filepath, responses_filepath, processed_indices_filepath, "https://api.openai.com/v1/chat/completions", api_key, max_requests_per_minute=5_000, max_tokens_per_minute=125_000, token_encoding_name="cl100k_base", max_attempts=3, logging_level=20))