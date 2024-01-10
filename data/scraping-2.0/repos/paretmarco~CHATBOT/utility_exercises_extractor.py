import openai
import re
import json
import os
import logging
import chardet

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    return chardet.detect(raw_data)['encoding']

def parse_instructions(prompt, text_chunk):
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts exercises, their goals, and names (if available) from given text."},
            {"role": "user", "content": f"{prompt}\n{text_chunk}"}
        ],
        temperature=0.3,
        max_tokens=900,
    )

    return completions.choices[0].message["content"].strip()

api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key

chunk_directory = "C:/D/documenti/AI/program24/chunker_split_exercises"
output_directory = "C:/D/documenti/AI/program24/chunk_extracted_exercises"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

prompt = "Can you identify an exercise or question in the following text, its goal, and its name (if available)? If not, reply 'FALSE'. If yes, reply 'TRUE' followed by the exercise, its goal, and its name (if available)."

for chunk_filename in os.listdir(chunk_directory):
    if not chunk_filename.endswith(".txt"):
        continue

    encoding = detect_encoding(os.path.join(chunk_directory, chunk_filename))
    print(f"Detected encoding for {chunk_filename}: {encoding}")

    with open(os.path.join(chunk_directory, chunk_filename), 'r', encoding=encoding) as f:
        text_chunk = f.read()

    logging.debug('Sending instructions to GPT-3.5-turbo...')
    parsed_instructions = parse_instructions(prompt, text_chunk)
    logging.debug('Parsed instructions received:')
    logging.debug(parsed_instructions)

    if parsed_instructions.startswith("TRUE"):
        exercise_data = parsed_instructions[5:].strip()

        # Extract exercise name or use a default name if not available
        exercise_name = "exercise.txt"
        exercise_name_start = exercise_data.find("Name:") + len("Name:")
        if exercise_name_start > len("Name:"):
            exercise_name_end = exercise_data.find(".", exercise_name_start)
            extracted_name = exercise_data[exercise_name_start:exercise_name_end].strip()
            if extracted_name != "Not available":
                exercise_name = extracted_name + ".txt"

        # Replace invalid characters in the exercise name
        exercise_name = re.sub(r'[\\/*?:"<>|\n]', '_', exercise_name)
 
        # Prepend the chunk name to the exercise name
        exercise_name = os.path.splitext(chunk_filename)[0] + "_" + exercise_name
        exercise_name = exercise_name[:100]

        # Save exercise data to a file with the exercise name
        with open(os.path.join(output_directory, exercise_name), 'w') as f:
            f.write(exercise_data)
    else:
        print(f"No exercise found in {chunk_filename}.")
