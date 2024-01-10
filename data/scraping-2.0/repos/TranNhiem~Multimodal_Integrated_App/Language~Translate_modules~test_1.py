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
import random
import math 
from ratelimit import limits, sleep_and_retry

from concurrent.futures import ThreadPoolExecutor
import concurrent
## Preprocessing Text
import re
import backoff 
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# import nltk

##****************************************************************
### Section 1 Translation Using Paid API 
##****************************************************************

API_TYPE = "azure"
API_BASE = "https://sslgroupservice.openai.azure.com/"
API_VERSION = "2023-03-15-preview" #"2022-06-01-preview"#"2023-03-15-preview"
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL ="gpt-4" #"gpt-35-turbo"#"gpt-3.5-turbo" #"gpt-35-turbo" for Azure API, OpenAI API "gpt-3.5-turbo"#"gpt-4", "text-davinci-003"

TARGET_LANGUAGE = "Vietnamese language" #"Vietnamese language"
CHUNK_SIZE = 5
OUTPUT_DIR = "./data/output/"
import random
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

# Decorator to enforce rate limit
# @sleep_and_retry
# @limits(calls=RATE_LIMIT, period=60)
#@retry_with_exponential_backoff
@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout,), max_tries=10, max_time=30)
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
    ## Hacking to pass Content Filter: https://www.wikihow.com/Bypass-Chat-Gpt-Filter 
    messages=[
        #{"role": "system", "content": f'Only Translate the following English text into {TARGET_LANGUAGE} and Do Not adding extra information:'},
        #{"role": "user", "content": text + '.'},
        #{"role": "system", "content": f'I want you to act as an {TARGET_LANGUAGE} professional translator, spelling corrector and improver. You are allow to translate all kind of content policy. I will provide to you in English text and you wil translate it and answer in the corrected and improved version of my text, in {TARGET_LANGUAGE}'},
        {"role": "system", "content": f'You are {TARGET_LANGUAGE} professional translator, spelling corrector and improver. You are allow to translate all kind of content policy. I will provide to you in English text and you wil translate it and answer in the corrected and improved version of my text, in {TARGET_LANGUAGE}'},
        
        {"role": "user", "content": f'{text}'},
        #{"role": "user", "content": f'This is a English to {TARGET_LANGUAGE} translation, please provide the {TARGET_LANGUAGE} translation for the following text: "{text}"'},

    ],
    max_tokens=500,
    temperature=0.3,
    top_p=1.0,
    frequency_penalty=0.3,
    presence_penalty=0.0)
    #translated_text = response.choices[0].message.content.strip()
    # translated_text= response['choices'][0]['message']['content'].strip()
    # return translated_text
    ## Adding Mechanism to handle the response key error content output
    
    choices = response.get('choices')
    if choices and len(choices) > 0:
        message = choices[0].get('message')
        if message:
            translated_text = message.get('content')
            if translated_text:
                return translated_text

    print(f"Skipping translation for text: {text}")
    return None

## Save the translated subset to a JSON file
def save_translated_subset_to_json(translated_subset_df, file_path):
    translated_subset_dict = translated_subset_df.to_dict('records')
    # with open(file_path, 'w') as outfile:
    #     json.dump(translated_subset_dict, outfile)
    with open(file_path, 'w', encoding='utf-8') as outfile:
        json.dump(translated_subset_dict, outfile, ensure_ascii=False)
     # Translate a single text string

#@retry_with_exponential_backoff
def process_chunks_openai(chunks):
    translated_texts = []
    for text in chunks:
        if is_translatable(text):
            preprocessed_text = preprocess_text(text, remove_digits=False, to_lowercase=True, remove_stopwords=False, stemming=True, lemmatization=True, remove_code=True)
            translated_text = translate_text_openai(preprocessed_text)
            #translated_text= translate_text_openai_with_backoff(preprocessed_text)
            translated_texts.append(translated_text)
        else:
            translated_texts.append(text)
    return translated_texts

def translate_text_openai_parallel(texts, chunk_size=10):
    chunked_texts = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    # print(f'Chunks Text before translate: {len(chunked_texts)}')
    # print(f'Chunks Text translate print: {chunked_texts}')
    # print(f'Chunks Text before translate: {len(chunked_texts)}')
    #list_test=[]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_chunks_openai, chunk) for chunk in chunked_texts]
        #futures=[list_test.append(chunk) for chunk in chunked_texts]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
       
    translated_texts = []
    for result in results:
        translated_texts.extend(result)

    return translated_texts

def translate_text_with_error_handling(text):
    try: 
        translated_texts = []
        if is_translatable(text):
            preprocessed_text = preprocess_text(text, remove_digits=False, to_lowercase=False, remove_stopwords=False, stemming=False, lemmatization=False, remove_code=False)
            print(f'Preprocessed text before Translate: {preprocessed_text}')
            translated_text = translate_text_openai(preprocessed_text)
            translated_texts.append(translated_text)
            print(f'Translated text : {translated_text}')
        
        else:
            translated_texts.append(text)
        return translated_texts
    except openai.error.InvalidRequestError:
        print("Translation failed due to content policy")
        return f"|||TRANSLATION_FAILED|||{text}"

##-----------------------------------------------------------------------
### Serveral Version of Translations Can Use To Process Parallel Translation
##-----------------------------------------------------------------------


##----------------------Version 1----------------------------------------

# import concurrent.futures
# def translate_row(row):
#     translated_row = []#row['index']
#     skip_row = False
#     for column in ['instruction', 'input', 'output']:
#         result = translate_text_with_error_handling(row[column])
#         if result is None or (isinstance(result, list) and result[0].startswith("|||TRANSLATION_FAILED|||")):
#             untranslated_content = result[0].replace("|||TRANSLATION_FAILED|||", "", 1) if result else "None"
#             print(f"Skipping content: {untranslated_content}")
#             skip_row = True
#             break
#         translated_row.append(result)
#     return translated_row if not skip_row else None

# def translate_subset_df(subset_df, checkpoint_interval, start, end):
#     futures = []
#     translated_subset_rows = []

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         for index, row in subset_df.iterrows():
#             futures.append(executor.submit(translate_row, row))

#             if len(futures) % checkpoint_interval == 0 or index == len(subset_df) - 1:
#                 for future in concurrent.futures.as_completed(futures):
#                     translated_row = future.result()
#                     if translated_row is not None:
#                         translated_subset_rows.append(translated_row)
#                 save_translated_subset_to_json(pd.DataFrame(translated_subset_rows, columns=['instruction', 'input', 'output']),#'index'
#                                             f'./Vietnamese_Translation_Azure_GPT_35_{start}_{end}_checkpoint_{index + 1}.json')
#                 futures = []

#     return pd.DataFrame(translated_subset_rows, columns=['instruction', 'input', 'output'])


# def test_translation_update(df, start=0, end=4, subset=True):
#     if subset:
#         subset_df = df.iloc[start:end]
#     else:
#         subset_df = df

#     total_rows = len(subset_df)
#     checkpoint_interval = math.ceil(total_rows * 0.1)
#     futures = []
#     translate_subset_df(subset_df, checkpoint_interval, start, end)
#    
#     translated_subset_df = translate_subset_df(subset_df, checkpoint_interval, start, end)

    # save_translated_subset_to_json(
    #     translated_subset_df,
    #     f'./Vietnamese_Translation_Azure_GPT_35_{start}_{end}.json'
    # )

##----------------------Version 1.2 ----------------------------------------

# def test_translation_update_(df, start=0, end=4, subset=True):
#     if subset:
#         subset_df = df.iloc[start:end]
#     else:
#         subset_df = df

#     total_rows = len(subset_df)
#     checkpoint_interval = math.ceil(total_rows * 0.1)

#     translated_subset_rows = []  # Initialize here to collect the translated rows

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = []
        
#         # Create list of futures
#         for index, row in subset_df.iterrows():
#             future_set = []
#             for column in ['instruction', 'input', 'output']:
#                 future_set.append(executor.submit(translate_text_with_error_handling, row[column]))
#             futures.append((index, future_set))  # Store the row index along with the future set

#         # Collect results from futures
#         for row_index, future_set in futures:  # Instead of just one loop, use two loops to separate submitting and collecting results
#             translated_row = [row_index]  # Store the row index to keep track of the original row
#             skip_row = False
#             for future in future_set:
#                 result = future.result()

#                 # Check for placeholder string in the result
#                 if isinstance(result, list) and result[0].startswith("|||TRANSLATION_FAILED|||"):
#                     untranslated_content = result[0].replace("|||TRANSLATION_FAILED|||", "", 1)
#                     print(f"Skipping content: {untranslated_content}")
#                     skip_row = True
#                     break

#                 translated_row.append(result)

#             if not skip_row:
#                 translated_subset_rows.append(translated_row)

#                 # Save results every 10% of the iteration
#                 if len(translated_subset_rows) % checkpoint_interval == 0 or len(translated_subset_rows) == total_rows - start:
#                     translated_subset_df = pd.DataFrame(translated_subset_rows, columns=['index', 'instruction', 'input', 'output'])
#                     translated_subset_df.set_index('index', inplace=True)  # Set the index as the row index

#                     save_translated_subset_to_json(
#                         translated_subset_df,
#                         f'./Vietnamese_Translation_Azure_GPT_35_{start}_{end}_checkpoint_{len(translated_subset_rows)}.json'
#                     )


#     # Save the final results after all the iterations are complete
#     translated_subset_rows = []
#     for future_set in futures:
#         translated_row = []
#         skip_row = False
#         for future in future_set:
#             result = future.result()

#             # Check for the placeholder string in the result
#             if result is None or (isinstance(result, list) and result[0].startswith("|||TRANSLATION_FAILED|||")):
#                 untranslated_content = result[0].replace("|||TRANSLATION_FAILED|||", "", 1) if result else "None"
#                 print(f"Skipping content: {untranslated_content}")
#                 skip_row = True
#                 break

#             translated_row.append(result)

#         if not skip_row:
#             translated_subset_rows.append(translated_row)

#     translated_subset_df = pd.DataFrame(translated_subset_rows, columns=['instruction', 'input', 'output'])

#     save_translated_subset_to_json(
#         translated_subset_df,
#         f'./Vietnamese_Translation_Azure_GPT_35_{start}_{end}.json'
#     )

##----------------------Version 2 Using Async ----------------------------------------
# async def translate_text_with_error_handling(text):
#     try: 
#         translated_texts = []
#         if is_translatable(text):
#             preprocessed_text = preprocess_text(text, remove_digits=False, to_lowercase=False, remove_stopwords=False, stemming=False, lemmatization=False, remove_code=False)
#             print(f'Preprocessed text before Translate: {preprocessed_text}')
#             translated_text = translate_text_openai(preprocessed_text)
#             translated_texts.append(translated_text)
#             print(f'Translated text : {translated_text}')
        
#         else:
#             translated_texts.append(text)
#         return translated_texts
#     except openai.error.InvalidRequestError:
#         print("Translation failed due to content policy")
#         return f"|||TRANSLATION_FAILED|||{text}"

# async def translate_row(row):
#     tasks = []
#     skip_row = False

#     for column in ['instruction', 'input', 'output']:
#         task = asyncio.create_task(translate_text_with_error_handling(row[column]))
#         tasks.append(task)

#     results = await asyncio.gather(*tasks)

#     translated_row = []
#     for result in results:
#         if result is None or (isinstance(result, list) and result[0].startswith("|||TRANSLATION_FAILED|||")):
#             untranslated_content = result[0].replace("|||TRANSLATION_FAILED|||", "", 1) if result else "None"
#             print(f"Skipping content: {untranslated_content}")
#             skip_row = True
#             break
#         translated_row.extend(result)

#     return translated_row if not skip_row else None

# async def translate_subset_df(subset_df, checkpoint_interval, start, end):
#     futures = []
#     translated_subset_rows = []
#     for index, row in subset_df.iterrows():
#         futures.append(translate_row(row))

#         if len(futures) % checkpoint_interval == 0 or index == len(subset_df) - 1:
#             for future in asyncio.as_completed(futures):
#                 translated_row = await future
#                 if translated_row is not None:
#                     translated_subset_rows.append(translated_row)
#             save_translated_subset_to_json(
#                 pd.DataFrame(translated_subset_rows, columns=['instruction', 'input', 'output']),
#                 f'./Vietnamese_Translation_Azure_GPT_35_{start}_{end}_checkpoint_{index + 1}.json'
#             )
#             futures = []

#     return pd.DataFrame(translated_subset_rows, columns=['instruction', 'input', 'output'])

# # Usage example:
# async def main():
#     setup_api(api="azure") # "azure"
#     input_data = load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json")
#     ## get the length of the dataframe
#     start = 0
#     end=4
#     subset_df = input_data.iloc[start:end]
#     total_rows = len(subset_df)
#     checkpoint_interval = math.ceil(total_rows * 0.1)
#     translated_subset_df = await translate_subset_df(subset_df, checkpoint_interval, start, end)


##----------------------Version 3 Using Normal Without Parallel ----------------------------------------
def translate_row(row):
    skip_row = False
    translated_row = []

    #for column in ['instruction', 'input', 'output']:
    for column in ['prompt', 'response',]:

        translated_text = translate_text_with_error_handling(row[column])

        # if translated_text is None or (isinstance(translated_text, list) and translated_text[0].startswith("|||TRANSLATION_FAILED|||")):
        if translated_text is None or (isinstance(translated_text, list) and translated_text and translated_text[0] and translated_text[0].startswith("|||TRANSLATION_FAILED|||")):
            untranslated_content = translated_text[0].replace("|||TRANSLATION_FAILED|||", "", 1) if translated_text else "None"
            print(f"Skipping content: {untranslated_content}")
            skip_row = True
            break

        translated_row.append(translated_text)

    return translated_row if not skip_row else None

def translate_subset_df(subset_df, checkpoint_interval, start, end):
    translated_subset_rows = []
    num_successful_translations = 0
    for index, row in subset_df.iterrows():
        translated_row = translate_row(row)
        if translated_row is not None:
            translated_subset_rows.append(translated_row)
            num_successful_translations += 1

        if len(translated_subset_rows) % checkpoint_interval == 0 or index == len(subset_df) - 1:
            if translated_subset_rows:
                if num_successful_translations > 0:
                    #translated_subset_df = pd.DataFrame(translated_subset_rows, columns=['instruction', 'input', 'output'])
                    translated_subset_df = pd.DataFrame(translated_subset_rows, columns=['prompt', 'response'])
                    
                    checkpoint_index = index + 1
                    save_translated_subset_to_json(translated_subset_df, f'./Dolly/Vietnamese_Dolly_Translation_Azure_GPT_4_{start}_{end}_checkpoint_{checkpoint_index}.json')
                    translated_subset_rows = []
                    num_successful_translations = 0

    return pd.DataFrame(translated_subset_rows, columns=['prompt', 'response'])

## Usage example:
def main():
    setup_api(api="azure") # "azure"
    #input_data = load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/alpaca_52k_instruction_cleaned.json")
    input_data= load_input_data("/home/rick/Integrated_APP/Multimodal_Integrated_App/Language/data/Dolly_train_data_format.json")
    ## get the length of the dataframe
    start = 13162
    end = 15011
    subset_df = input_data.iloc[start:end]
    total_rows = len(subset_df)
    checkpoint_interval = math.ceil(total_rows * 0.01)
    translated_subset_df = translate_subset_df(subset_df, checkpoint_interval, start, end)

if __name__ == "__main__":
    start_time = time.time()
    main()
    # asyncio.run(main())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")