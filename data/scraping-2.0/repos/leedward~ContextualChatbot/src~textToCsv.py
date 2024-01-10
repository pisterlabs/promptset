import os
import re
import time
import requests
import openai
import chardet
import pandas as pd
import tiktoken
from dotenv import load_dotenv

# Settings
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
TEXT_FOLDER = 'data/text'
PROCESSED_FOLDER = 'data/text-processed'
SCRAPE_FOLDER = 'data/processed/scraped'
TOKENIZER_NAME = "cl100k_base"
MAX_TOKENS = 250
MIN_TOKENS = 100
MAX_RETRIES = 3
SLEEP_TIME = 60
MODEL_NAME = "gpt-3.5-turbo"

# Load the tokenizer
tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)


def split_into_chunks(text: str, max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS) -> list:
    sentences = text.split('. ')
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, tokens in zip(sentences, n_tokens):
        if tokens_so_far + tokens > max_tokens:
            if len(chunk) > 0:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0
        if tokens > max_tokens:
            continue
        chunk.append(sentence)
        tokens_so_far += tokens

    if len(chunk) > 0:
        if len(chunks) > 0 and len(tokenizer.encode(". ".join(chunk))) < min_tokens:
            chunks[-1] += ". " + ". ".join(chunk)
        else:
            chunks.append(". ".join(chunk) + ".")

    print(f"Total chunks to be processed: {len(chunks)}")
    return chunks


def preprocess_and_extract_info(text: str) -> str:
    text = re.sub('\s+', ' ', text.replace('\n', ' ').replace('\\n', ' ').replace('  ', ' '))
    chunks = split_into_chunks(text)
    extracted_texts = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk number {i + 1} out of {len(chunks)}")
        for _ in range(MAX_RETRIES):
            try:
                response = openai.ChatCompletion.create(
                    model=MODEL_NAME,
                    temperature=0.8,
                    messages=[
                        {"role": "system",
                         "content": "Act as an advanced REGEX algorithm. Your task is to extract all relevant information from a {Text} that is related to the provided {Context} by removing all the elements you would classify in the {RemoveContext}. The goal is to preserve as much of the original {Text} as possible, organised and coherent, with relevant information presented in a logical sequence ensuring you maintain the original language and style as much as possible, but without the aforementioned unrelated content that matches {RemoveContext}. The response should be presented in a clear and concise manner as close to the original {Text} as possible, with all formatting and placeholder characters (e.g. newline, tab, carriage return) removed. If there is no relevant information related to the {Context} in the provided {Text}, return \"No relevant content\". Step 1 is removing unrelated content from the {Text} based on the {RemoveContext}. Step 2 is to return as much of the remaining original information from the {Text}, that is relevant to {Context}, in a coherent structure."},
                        {"role": "user",
                         "content": "Context: the subject of Cryptocurrency, encapsulating specifics of various digital currencies, blockchain technology and software development, cryptographic measures, transaction protocols, mining techniques or DeFi applications. Also include trading platforms, tools for financial analysis or security, or usage of digital and hardware wallets. Any security protocols ensuring safe crypto transactions or messaging. Furthermore, instructional guides, case studies, or insightful documentation"},
                        {"role": "user",
                         "content": "RemoveContext: terms and conditions, privacy policies, legal disclaimers, liability statements, purposes of the information, endorsements, author names, web page navigation text, any irrelevant promotional content, or non-informative asides such as author's opinions, site endorsements or advice."},
                        {"role": "user", "content": f"Text: {chunk}"}
                    ]
                )

                content = response['choices'][0]['message']['content'].strip()
                if "No relevant content" not in content:
                    extracted_texts.append(content.replace('\n', ' ').replace('\\n', ' ').replace('  ', ' '))
                else:
                    print("Chunk contains no valid information. Skipping chunk...")
                break
            except (openai.error.RateLimitError, openai.error.APIError, requests.exceptions.ReadTimeout,
                    openai.error.APIConnectionError) as e:
                print(f"An error occurred: {str(e)}. Retrying in {SLEEP_TIME} seconds...")
                time.sleep(SLEEP_TIME)
            except Exception as e:
                print(f"An unexpected error occurred: {str(e)}. Retrying in {SLEEP_TIME} seconds...")
                time.sleep(SLEEP_TIME)

    return " ".join(extracted_texts)


def process_file(domain: str, file: str, output_file: str, first_file: bool) -> None:
    print(f"Processing file: {file}")
    if os.path.isfile(os.path.join(TEXT_FOLDER, domain, file)):
        rawdata = open(os.path.join(TEXT_FOLDER, domain, file), 'rb').read()
        result = chardet.detect(rawdata)
        encoding = result['encoding']
        print(f"Detected encoding for file '{file}': {encoding}")

        with open(os.path.join(TEXT_FOLDER, domain, file), "r", encoding=encoding, errors='ignore') as f:
            text = f.read()
            fname = file.replace(f"{domain}_", '').replace('_', ' ').replace('-', ' ')[:-4]
            df = pd.DataFrame([(fname, text)], columns=['fname', 'text'])
            df['text'] = df.text.apply(preprocess_and_extract_info)

            if pd.notnull(df['text']).all() and df['text'].str.strip().ne("").all():
                df.to_csv(output_file, mode='a', index=False, header=first_file)


def poll_and_process_domain(domain: str) -> None:
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    os.makedirs(SCRAPE_FOLDER, exist_ok=True)

    processed_files = set()  # Set to keep track of processed files
    output_file = f'{SCRAPE_FOLDER}/{domain}_scraped.csv'
    first_file = True

    while True:  # Loop indefinitely
        files = os.listdir(f"{TEXT_FOLDER}/{domain}/")
        new_files = [file for file in files if file not in processed_files]  # Check for new files

        for file in new_files:
            process_file(domain, file, output_file, first_file)
            first_file = False
            processed_files.add(file)  # Mark as processed

        if new_files:
            print("Processed new files. Waiting for more...")
        else:
            print(f"No new files. Sleeping for {SLEEP_TIME} seconds...")

        time.sleep(SLEEP_TIME)  # Wait for 60 seconds


if __name__ == "__main__":
    poll_and_process_domain('your_domain')
