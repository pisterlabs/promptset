#####################
#     Libraries     #
#####################

import openai
from dotenv import load_dotenv
import os
from tqdm import tqdm
import pandas as pd

# Set the OpenAI API key
def set_api_key():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


###################################
#    PARAPHRASING W. CHAT GPT     #
###################################

# Paraphrase a list of strings and output a list of paraphrased strings (using gpt-3):
def paraphrase_text_list(text_list: list) -> list:
    """
    Takes a list of strings and paraphrases them using the gpt-3 chat model.
    
    text_list: list of strings to paraphrase
    """
    paraphrased_list = []
    print("Paraphrasing...")
    for text in tqdm(text_list):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Your task is to proficiently understand and communicate in Danish. You are required to rephrase text in Danish while adhering to the following rules:

1. Avoid repeating yourself.
2. Refrain from using the same sentence as in the original text.
3. Maintain a similar text length to the original.
4. Ensure the context remains consistent with the original text.
5. Be creative in your rephrasing.

Please provide your rephrased response in Danish, observing the given rules and maintaining the context of the original text."""},
                {"role": "user", "content": text},
            ]
        )
        paraphrased_text = response['choices'][0]['message']['content']
        paraphrased_list.append(paraphrased_text)
    return paraphrased_list

def df_with_original_and_paraphrased_text(text_list: list, paraphrased_list: list) -> list:
    """
    Takes a list of strings and a list of paraphrased strings and outputs a dataframe with the original and paraphrased text.
    
    text_list: list of strings to paraphrase
    paraphrased_list: list of paraphrased strings
    """
    df = pd.DataFrame(list(zip(text_list, paraphrased_list)), columns =['original_text', 'paraphrased_text'])
    return df

def semantic_similarity(df: pd.DataFrame) -> list:
    """
    Takes a dataframe with original and paraphrased text and outputs a list of semantic similarity scores.
    
    df: dataframe with original and paraphrased text
    """
    similarity = []
    print("Calculating semantic similarity...")
    for i in tqdm(range(len(df))):
        response = openai.Engine("davinci").search(
            documents=[df['original_text'][i]],
            query=df['paraphrased_text'][i],
            max_rerank=10,
            return_metadata=True,
        )
        similarity.append(response['data'][0]['score'])
    return similarity