import os
import re
from dtu_api import API_KEY
import openai
import random
openai.api_key = API_KEY
from tqdm import tqdm
import tiktoken
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
from retry import retry
import time

from concurrent.futures import ThreadPoolExecutor

def count_gpt35_tokens(text):
    tokens = tokenizer.encode_ordinary(text)
    return len(tokens)

def split_paragraphs(text):
    return re.split(r'\n+', text.strip())

def create_chunks(filename, max_tokens=2500):
    with open(filename, 'r', encoding='iso-8859-1') as file:
        content = file.read()

    paragraphs = split_paragraphs(content)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = count_gpt35_tokens(paragraph)

        if paragraph_tokens > max_tokens:
            #print(f"Skipping file {filename} due to paragraph exceeding max tokens, which indicates an error.")
            #write an empty file to bad_files
            basename = os.path.basename(filename)
            with open(os.path.join(bad_files, basename), 'w', encoding='utf-8') as bad_file:
                bad_file.write('')
            return []  # Return an empty list to indicate skipping the file

        if current_tokens + paragraph_tokens <= max_tokens:
            current_chunk.append(paragraph)
            current_tokens += paragraph_tokens
        else:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_tokens = paragraph_tokens

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks




#adding exponential retry
@retry(tries=2, delay=1, backoff=2)
def process_file(file_info):
    input_folder, output_folder, filename = file_info
    if filename.endswith('.txt'):
        file_path = os.path.join(input_folder, filename)
        chunks = create_chunks(file_path)

        for i, chunk in enumerate(chunks):
            retry_count = 0
            while retry_count < 2:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": """Læs teksten nøje og besvar mindst 5 faktuelle spørgsmål, der kan opstå ud fra teksten. Hvert spørgsmål skal indeholde nok kontekst til at være forståeligt uden at have set teksten. Svar på spørgsmålene ved at udtrække relevant information fra teksten med fokus på konkrete detaljer, der har relevans for spørgsmålene.

Angiv dit svar således:

Spørgsmål 1: [indsæt]
Svar 1: [indsæt]

Spørgsmål 2: [indsæt]
Svar 2: [indsæt]

Spørgsmål 3: ..."""
                            },
                            {"role": "user", "content": chunk},
                        ],
                        temperature=0.6,
                        max_tokens=1000,
                    )

                    response_text = response.choices[0].message['content']

                    output_filename = f"{os.path.splitext(filename)[0]}_{i}.txt"
                    output_path = os.path.join(output_folder, output_filename)
                    with open(output_path, 'w', encoding='utf-8') as output_file:
                        output_file.write(response_text.strip())
                    break  # Successfully processed the chunk, exit the loop

                except Exception as e:
                    print(f"Error processing chunk {i} of file {filename}: {e}")
                    if "Rate limit" in str(e):
                        # If rate limit reached, wait for 60 seconds and try again
                        time.sleep(60)
                    retry_count += 1

            if retry_count == 2:  # If the second attempt fails, skip the chunk
                print(f"Skipping chunk {i} of file {filename} after 2 failed attempts")

def process_files(input_folder, output_folder, workers=20):
    os.makedirs(output_folder, exist_ok=True)
    files = os.listdir(input_folder)

    # Get a set of basenames for the processed files
    processed_files = {os.path.splitext(entry.name)[0].rsplit('_', 1)[0] for entry in os.scandir(output_folder) if entry.is_file()}

    # Filter out files that have already been processed
    file_infos = [(input_folder, output_folder, f) for f in files if os.path.splitext(f)[0] not in processed_files]

    # Shuffle the list of files to process
    random.shuffle(file_infos)

    # Print number of files that will be skipped
    print(f"Skipping {len(files) - len(file_infos)} files that have already been processed.")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(process_file, file_infos), total=len(file_infos), desc="Processing files"))


if __name__ == '__main__':
    input_folder = 'data/all_paragraphs_large_removed/paragraphs'
    output_folder = 'data/output_responses'
    bad_files = 'data/bad_files'
    os.makedirs(bad_files, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
    process_files(input_folder, output_folder)
