'''
@TranNhiem 2023/05
This design including 2 Sections:

1. Using The Pay API to Translate Dataset
    + OpenAI API (gpt-3.5-turbo) & GPT-3 API (text-davinci-003)
    + Azure Translate Optional 
    + Google Translation API Optional 
2. Using Open-Source Pretrained Language Model for Transslation 
    + NLLB - MetaAI Translation Model 
    + BLOOM - Opensource Multi-lingual Language Model
    + T5&FlanT5 - Google's Text-to-Text 
'''

import os
import openai
import json
import pandas as pd
import numpy as np
import re
import glob
import torch
import string
import requests
import time 

from ratelimit import limits, sleep_and_retry

from concurrent.futures import ThreadPoolExecutor
import concurrent
## Preprocessing Text
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
#nltk.data.path.append('/home/rick/nltk_data')
# nltk.download('stopwords')  # Add this line to download stopwords explicitly
# nltk.download('wordnet')
# nltk.data.path.append('/path/to/nltk_data')

# nltk.download()
#nltk.download("wordnet", force=True)

##  Download Original English Dataset Version
#url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
# url="https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
# url="https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/main/data/alpaca_gpt4_data.json"
# response = requests.get(url)
# output_file = "./data/alpaca_52k_GPT4_instruction_cleaned.json"
# if response.status_code == 200:
#     with open(output_file, 'wb') as f:
#         f.write(response.content)
#     print(f"File downloaded successfully and saved as {output_file}")
# else:
#     print(f"Failed to download the file. Status code: {response.status_code}")
# Constants


##****************************************************************
### Section 1 Translation Using Paid API 
##****************************************************************

API_TYPE = "azure"
API_BASE = "https://sslgroupservice.openai.azure.com/"
API_VERSION = "2023-03-15-preview" #"2022-06-01-preview"#"2023-03-15-preview"
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-35-turbo"#"gpt-3.5-turbo" #"gpt-35-turbo" for Azure API, OpenAI API "gpt-3.5-turbo"#"gpt-4", "text-davinci-003"

TARGET_LANGUAGE = "Traditonal Chinese language" #"Vietnamese language"
CHUNK_SIZE = 5
OUTPUT_DIR = "./data/output/"

# Set up API
def setup_api(api="azure"):
    if api == "azure":
        openai.api_type = API_TYPE
        openai.api_base = API_BASE
        openai.api_version = API_VERSION
        openai.api_key = API_KEY
    else:
        openai.organization = "org-PVVobcsgsTm9RT8Ez5DubzbX" # Central IT account
        #openai.api_key = API_KEY
        openai.api_key = os.getenv("OPENAI_API_KEY")

# Load input data as DataFrame
def load_input_data(INPUT_TASKS_PATH):
    with open(INPUT_TASKS_PATH, "rb") as f:
        json_data = json.loads(f.read())
    return pd.DataFrame(json_data)

# Save the given data to a JSON file at the specified file path
def write_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

##----------- Start PREPROCESSING TEXT ------------------------

# def remove_urls(text):
#     url_pattern = re.compile(r'https?://\S+|www\.\S+')
#     return url_pattern.sub(r'', text)

# def remove_html_tags(text):
#     html_pattern = re.compile(r'<.*?>')
#     return html_pattern.sub(r'', text)

# def remove_special_characters(text, keep_chars="'.,!?"):
#     pattern = re.compile(f'[^A-Za-z0-9{keep_chars}\s]')
#     return pattern.sub(r'', text)

# def matches_regex(regex, text):
#     return bool(re.compile(regex).search(text))

# def contains_code(text):
#     code_blacklist = ['&&', '||', '<html>', ';\n', 'SELECT']
#     return (
#         any(code_keyword in text for code_keyword in code_blacklist) or
#         matches_regex(r'\w+\(\w*\) \{', text) or
#         matches_regex(r'def \w+\(', text) or
#         matches_regex(r'\[A-z]+\.[A-z]+', text) or
#         matches_regex(r': [\w\.#]{1,12};', text) or
#         matches_regex(r'<\/\w+>', text)
#     )

# def preprocess_text(text, remove_digits=False, to_lowercase=False, remove_stopwords=False, stemming=False, lemmatization=False, keep_chars="'.,!?", remove_code=False):
    
#     def remove_punctuation(text):
#         return ''.join(c if c not in string.punctuation or c == '-' else ' ' for c in text)
  
#     # Remove URLs
#     text = remove_urls(text)

#     # Remove HTML tags
#     text = remove_html_tags(text)

#     # Remove special characters
#     text = remove_special_characters(text, keep_chars=keep_chars)

#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text).strip()

#     # Remove code content
#     if remove_code:
#         text = re.sub(r'(?s)(?P<tag><code>.*?</code>)', '', text)

#     if remove_digits:
#         text = re.sub(r'\d+', '', text)

#     if to_lowercase:
#         text = text.lower()
#     # Call the remove_punctuation function
#     text = remove_punctuation(text)
   
    
#     if remove_stopwords or stemming or lemmatization:
#         tokens = word_tokenize(text)
#         if remove_stopwords:
#             #stop_words = set(stopwords.words('english'))
#             stop_words = set(stopwords.words('english')).union(set(stopwords.words('english')))
#             text = " ".join([word for word in text.split() if word not in stop_words])
#         if stemming:
#             stemmer = PorterStemmer()
#             tokens = [stemmer.stem(token) for token in tokens]

#         if lemmatization:
#             lemmatizer = WordNetLemmatizer()
#             tokens = [lemmatizer.lemmatize(token) for token in tokens]

#         text = ' '.join(tokens)

#     return text

# # Check if the given text contains words
# def contains_words(text):
#     return matches_regex(r'[A-z]{3,}', text)

# # Check if the given text is translatable
# def is_translatable(text):
#     if text == "":
#         return False
#     return (contains_code(text) is False) and contains_words(text)

import spacy

# Load the SpaCy English language model
nlp = spacy.load("en_core_web_sm")

def remove_urls(text):
    # Implementation using SpaCy
    doc = nlp(text)
    text_without_urls = " ".join([token.text for token in doc if not token.like_url])
    return text_without_urls

def remove_html_tags(text):
    # Implementation using regular expressions
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub(r'', text)

def matches_regex(regex, text):
    return bool(re.compile(regex).search(text))
def remove_special_characters(text, keep_chars="'.,!?"):
    pattern = re.compile(f'[^A-Za-z0-9{keep_chars}\s]')
    return pattern.sub(r'', text)

def contains_code(text):
    code_blacklist = ['&&', '||', '<html>', ';\n', 'SELECT']
    return (
        any(code_keyword in text for code_keyword in code_blacklist) or
        matches_regex(r'\w+\(\w*\) \{', text) or
        matches_regex(r'def \w+\(', text) or
        matches_regex(r'\[A-z]+\.[A-z]+', text) or
        matches_regex(r': [\w\.#]{1,12};', text) or
        matches_regex(r'<\/\w+>', text)
    )

def preprocess_text(text, remove_digits=False, to_lowercase=False, remove_stopwords=False, stemming=False, lemmatization=False, keep_chars="'.,!?", remove_code=False):
    
    def remove_punctuation(text):
        return ''.join(c if c not in string.punctuation or c == '-' else ' ' for c in text)

    # Remove URLs using SpaCy
    text = remove_urls(text)

    # Remove HTML tags
    text = remove_html_tags(text)

    # Remove special characters
    text = remove_special_characters(text, keep_chars=keep_chars)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove code content
    if remove_code:
        text = re.sub(r'(?s)(?P<tag><code>.*?</code>)', '', text)

    if remove_digits:
        text = re.sub(r'\d+', '', text)

    if to_lowercase:
        text = text.lower()

    # Call the remove_punctuation function
    text = remove_punctuation(text)

    if remove_stopwords or stemming or lemmatization:
        # Tokenize the text using SpaCy
        doc = nlp(text)

        if remove_stopwords:
            # Remove stop words using SpaCy
            tokens = [token.text for token in doc if not token.is_stop]
        else:
            tokens = [token.text for token in doc]

        if stemming:
            # Perform stemming using SpaCy's Lemmatizer
            tokens = [token.lemma_ for token in doc]

        if lemmatization:
            # Perform lemmatization using SpaCy's Lemmatizer
            tokens = [token.lemma_ for token in doc]

        text = ' '.join(tokens)

    return text

# Check if the given text contains words
def contains_words(text):
    return matches_regex(r'[A-z]{3,}', text)

# Check if the given text is translatable
def is_translatable(text):
    if text == "":
        return False
    return (contains_code(text) is False) and contains_words(text)

##-----------END  PREPROCESSING TEXT ------------------------
# Delay between API calls to stay within rate limits
def delay_between_requests():
    time.sleep(7)  # Adjust the delay time as needed



# Define the rate limit (requests per minute) and token limit
## GPT3
# RATE_LIMIT = 120  # Adjust the rate limit as per your model
# TOKEN_LIMIT = 40000  # Adjust the token limit as per your model
## ChatGPT
RATE_LIMIT = 200
TOKEN_LIMIT = 120000

# ##GPT4
# RATE_LIMIT = 18
# TOKEN_LIMIT = 600000


# Decorator to enforce rate limit
@sleep_and_retry
@limits(calls=RATE_LIMIT, period=60)
def translate_text_openai(text):
    #delay_between_requests()  # Add delay before each API call
    if not text.strip():
        return ""
    # if ' ' in text:
    #     prompt= f'Please provide the {TARGET_LANGUAGE} translation for these sentences: {text}'
    # else:
    #     prompt= f'Please provide the {TARGET_LANGUAGE} translation for the following word: {text}'
    #prompt = f"Please translate the following English text to {TARGET_LANGUAGE} : {text}"
    # prompt= f" English text: {text} translation into Traditional Chinese language: " # Not greate result
    # prompt= f"Translate the following English text to Traditional language: {text}"
    # prompt= f'Please provide the {TARGET_LANGUAGE} translation for these sentences: {text}'
    # prompt = f'Translate the following English text into {TARGET_LANGUAGE}: "{text}"'
    
    # response = openai.Completion.create(
    #     engine=MODEL, 
    #     prompt=prompt, 
    #     max_tokens=800, 
    #     stop=None, 
    #     temperature=0.01,
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0
    # )
    # translated_text = response.choices[0].text.strip()
    # return translated_text.split('\n')[-1].strip()
    response = openai.ChatCompletion.create(
    engine=MODEL,
    messages=[
        {"role": "system", "content": f'Translate the following English text into {TARGET_LANGUAGE}:'},
        {"role": "user", "content": text}
    ],
    max_tokens=800,
    temperature=0.3,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0)
    translated_text = response.choices[0].message.content.strip()
    return translated_text


## Save the translated subset to a JSON file
def save_translated_subset_to_json(translated_subset_df, file_path):
    translated_subset_dict = translated_subset_df.to_dict('records')
    # with open(file_path, 'w') as outfile:
    #     json.dump(translated_subset_dict, outfile)
    with open(file_path, 'w', encoding='utf-8') as outfile:
        json.dump(translated_subset_dict, outfile, ensure_ascii=False)
     # Translate a single text string

def translate_text(text):
    if is_translatable(text):
        preprocessed_text = preprocess_text(text, remove_digits=False, to_lowercase=True, remove_stopwords=False, stemming=True, lemmatization=False, remove_code=True)
        #print(f"Text Before Translation: {text}")
        translated_text = translate_text_openai(preprocessed_text)
        #print(f"Text After Translation: {translated_text}")
        return translated_text
    else:
        return text

# Save the translated subset to a JSON file
def test_translation(df, start=0,end=4, subset=True):
    if subset:
        #subset_df = df.head(n_rows)
        subset_df= df.iloc[start:end]
    else:
        subset_df = df

    translated_instruction = subset_df['instruction'].apply(translate_text)
    translated_input = subset_df['input'].apply(translate_text)
    translated_output = subset_df['output'].apply(translate_text)

    translated_subset_df = pd.DataFrame({'instruction': translated_instruction, 
                                         'input': translated_input, 
                                         'output': translated_output})
    
    save_translated_subset_to_json(translated_subset_df, './data/output/translated_Traditional_Chinese_GPT_35_2023_newprompt3.json')

    # print("\nOriginal subset:")
    # print(subset_df)
    # print("\nTranslated subset:")
    # print(translated_subset_df)

### Update The Code To Process Text into Chunk & also Using Multi-Thread 
def process_chunks_openai(chunks):
    translated_texts = []
    for text in chunks:
        if is_translatable(text):
            preprocessed_text = preprocess_text(text, remove_digits=False, to_lowercase=True, remove_stopwords=False, stemming=True, lemmatization=True, remove_code=True)
            translated_text = translate_text_openai(preprocessed_text)
            translated_texts.append(translated_text)
        else:
            translated_texts.append(text)
    return translated_texts


def translate_text_openai_parallel(texts, chunk_size=100):
    chunked_texts = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    # print(f'Chunks Text before translate: {len(chunked_texts)}')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunks_openai, chunk) for chunk in chunked_texts]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    translated_texts = []
    for result in results:
        translated_texts.extend(result)

    return translated_texts

# Update the test_translation function to use translate_text_openai_parallel
def test_translation_update(df, start=0, end=4, subset=True):
    if subset:
        subset_df = df.iloc[start:end]
    else:
        subset_df = df

    translated_instructions = translate_text_openai_parallel(subset_df['instruction'].tolist())
    translated_inputs = translate_text_openai_parallel(subset_df['input'].tolist())
    translated_outputs = translate_text_openai_parallel(subset_df['output'].tolist())

    translated_subset_df = pd.DataFrame({'instruction': translated_instructions,
                                         'input': translated_inputs,
                                         'output': translated_outputs})

    save_translated_subset_to_json(translated_subset_df, './data/output/Tranditional_Chinese_Translation_Azure_GPT_35_10_20K.json')

def main():
        setup_api(api="azure") # "azure"
        input_data = load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json")
        ## get the length of the dataframe
        # df_length = len(input_data)
        # # print the length
        ## Old Version 
        #test_translation(input_data, start=0,end=10000, subset=True)
        # print(f"The length of the dataframe is: {df_length}")
        test_translation_update(input_data, start=10000,end=20000, subset=True)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")