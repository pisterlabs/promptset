# I understand that you want to rewrite the text in a more book-style format, with better linguistic form, and incorporate GPT-3.5-turbo to improve the content. You also want to use the previous chunk's text (if available) to help generate a more coherent output. I have modified your provided code to implement this feature. this is the old scripts that in any case worked


import os
import re
import json
import openai
import logging
from langdetect import detect
from langcodes import Language
from nltk.tokenize import sent_tokenize
from transformers import GPT2Tokenizer
from retry import retry

logging.basicConfig(level=logging.INFO)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

@retry(tries=5, delay=60, backoff=2)
def generate_text(prompt, model="gpt-3.5-turbo", temperature=0.8):
    max_length = 4096 - len(tokenizer.encode("[ChatGPT] ")) - 2
    prompt_tokens = tokenizer.encode(prompt)
    truncated_prompt_tokens = prompt_tokens[:max_length]
    truncated_prompt = tokenizer.decode(truncated_prompt_tokens)

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Generate relevant and creative text based on the given input."},
                {"role": "user", "content": truncated_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message['content'].strip()
    except openai.error.RateLimitError as e:
        logging.warning("Rate limit error, retrying: " + str(e))
        raise
    except Exception as e:
        logging.error("Unexpected error: " + str(e))
        return "Error: Unable to generate text."

input_dir = "C:/D/documenti/AI/program24/chunker_split_transcripts"
output_dir = "C:/D/documenti/AI/program24/chunker_processed_transcripts"

# Create output_dir if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for input_filename in os.listdir(input_dir):
    logging.info(f"Processing {input_filename}")

    if not input_filename.endswith(".txt"):
        continue

    input_path = os.path.join(input_dir, input_filename)
    with open(input_path, "r", encoding="utf-8") as f:
        chunk = f.read()

    # Detect language from the first 100 characters of the chunk
    first_characters = chunk[:100]
    language_code = detect(first_characters)
    language_name = Language.make(language=language_code).display_name()  # Convert language code to full language name
    logging.info(f"Detected language: {language_name}")  # Log detected language

    chunk_number = int(re.search(r"_part_(\d+)", input_filename).group(1))

    prev_chunk_number = chunk_number - 1
    prev_chunk_filename = re.sub(r"_part_(\d+)", f"_part_{prev_chunk_number}", input_filename)
    prev_chunk_path = os.path.join(output_dir, prev_chunk_filename.replace(".txt", "_processed.txt"))

    if os.path.exists(prev_chunk_path):
        with open(prev_chunk_path, "r", encoding="utf-8") as f:
            prev_chunk = f.read()
    else:
        prev_chunk = ""

    prompt = f"Write in the following language: ## {language_name} ##. Use Victor Hugo Style. Continue smoothly this text: ###{prev_chunk}### using, connecting, correcting and making understable this text: ###{chunk}###. Create long paragraphs with their own title and put the maximum of details. You write in a understandable format in the following language: ## {language_name} ##."
    improved_chunk = generate_text(prompt)

    book_name = os.path.splitext(input_filename)[0]
    output_filename = f"{book_name}_processed.txt"
    output_path = os.path.join(output_dir, output_filename)

    logging.info("Writing output file")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(improved_chunk)

    logging.info(f"Finished processing {input_filename}")
