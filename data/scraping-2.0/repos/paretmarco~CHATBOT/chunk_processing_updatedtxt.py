import os
import re
import json
import openai
import logging
from nltk.tokenize import sent_tokenize
from transformers import GPT2Tokenizer
from retry import retry

logging.basicConfig(level=logging.INFO)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

@retry(tries=5, delay=60, backoff=2)
def generate_text(prompt, model="gpt-3.5-turbo"):
    max_length = 4096 - len(tokenizer.encode("[ChatGPT] ")) - 2
    prompt_tokens = tokenizer.encode(prompt)
    truncated_prompt_tokens = prompt_tokens[:max_length]
    truncated_prompt = tokenizer.decode(truncated_prompt_tokens)

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Generate relevant and creative text based on the given input. You put relevant infos in order to better understand and index the text."},
                {"role": "user", "content": truncated_prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except openai.error.RateLimitError as e:
        logging.warning("Rate limit error, retrying: " + str(e))
        raise
    except Exception as e:
        logging.error("Unexpected error: " + str(e))
        return "Error: Unable to generate text."

input_dir = "C:/D/documenti/AI/program24/chunker_split"
output_dir = "C:/D/documenti/AI/program24/chunker_processed"

# Define the number of sentences to consider from the previous and next chunks
prev_context_sentences = 10
next_context_sentences = 10

for input_filename in os.listdir(input_dir):
    logging.info(f"Processing {input_filename}")

    if not input_filename.endswith(".txt"):
        continue

    input_path = os.path.join(input_dir, input_filename)
    with open(input_path, "r", encoding="utf-8") as f:
        chunk = f.read()

    chunk_number = int(re.search(r"_part_(\d+)", input_filename).group(1))

    # Get the last 10 sentences from the previous 10 chunks
    prev_chunk_numbers = [chunk_number - i for i in range(1, 11)]
    last_n_sentences = ""
    for prev_chunk_number in prev_chunk_numbers:
        prev_chunk_filename = re.sub(r"_part_(\d+)", f"_part_{prev_chunk_number}", input_filename)
        if os.path.exists(os.path.join(input_dir, prev_chunk_filename)):
            with open(os.path.join(input_dir, prev_chunk_filename), "r", encoding="utf-8") as f:
                prev_chunk = f.read()
            prev_sentences = sent_tokenize(prev_chunk)
            last_n_sentences += ' '.join(prev_sentences[-prev_context_sentences:]) + ' '

    # Get the first 10 sentences from the next 10 chunks
    # Get the first 10 sentences from the next 10 chunks
    next_chunk_numbers = [chunk_number + i for i in range(1, 11)]
    next_n_sentences = ""
    for next_chunk_number in next_chunk_numbers:
        next_chunk_filename = re.sub(r"_part_(\d+)", f"_part_{next_chunk_number}", input_filename)
        if os.path.exists(os.path.join(input_dir, next_chunk_filename)):
            with open(os.path.join(input_dir, next_chunk_filename), "r", encoding="utf-8") as f:
                next_chunk = f.read()
            next_sentences = sent_tokenize(next_chunk)
            next_n_sentences += ' '.join(next_sentences[:next_context_sentences]) + ' '

    logging.info("Generating title")
    title_input = f"Create a title in form of a question for this text of no more than 15 words in order to make clear the meaning of the text and to give important keywords: {last_n_sentences} {chunk} {next_n_sentences}"
    title_text = generate_text(title_input)

    logging.info("Generating first phrase")
    first_phrase_input = f"Speaking as you were the author, create one introductory text for the last part (200 tokens) of this text of no more than 70 words. This text should give a clear meaning and context to the last 200 token, give the relevant names of the people and concepts in it and make easy to situate. Give also with it a short summary of the tokens before. : {last_n_sentences} {chunk}"
    first_phrase_text = generate_text(first_phrase_input)

    logging.info("Generating last phrase")
    last_phrase_input = f"Create a concluding phrase for this text of no more than 70 words. This phrase should be useful for the reader and the most possible practical. If the text is an exercise write the completion and the steps using also these text: {chunk} {next_n_sentences}"
    last_phrase_text = generate_text(last_phrase_input)

    book_name = os.path.splitext(input_filename)[0]
    output_filename = f"{book_name}_processed.txt"
    output_path = os.path.join(output_dir, output_filename)

    logging.info("Writing output file")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "title": title_text,
            "first_phrase": first_phrase_text,
            "content": chunk,
            "last_phrase": last_phrase_text
        }, f, ensure_ascii=False, indent=2)

    logging.info(f"Finished processing {input_filename}")
