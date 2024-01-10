import os
import time
import openai
from dotenv import load_dotenv
import tiktoken

# Function to count tokens using tiktoken
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = sum(1 for _ in encoding.encode(text))
    return token_count

# Function to read and chunk the text file
def read_and_chunk_txt(file_path):
    chunks = []
    chunk = ""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if count_tokens(chunk + text) > 4000:
                chunks.append(chunk.strip())
                chunk = text
            else:
                chunk += " " + text
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Load environment variables for OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Read and chunk the text from the txt file
chunks = read_and_chunk_txt("data/raw_train.txt")

# Ask the user how many chunks they want to process
while True:
    process_option = input("Do you want to process one chunk (type 'one') or all chunks (type 'all')? ").strip().lower()
    if process_option in ['one', 'all']:
        break
    else:
        print("Invalid option. Please enter 'one' or 'all'.")

# Open output file
with open("data/train.txt", "w", encoding='utf-8') as train_file:
    for snippet in ["Clean up the above information and respond with the complete text. Avoid using special characters. Keep formatting very simple and suitable for a .txt file. Do not respond in markdown."]:
        for idx, chunk in enumerate(chunks):
            if process_option == 'one' and idx > 0:
                break

            prompt = f"{chunk}\n\n{snippet}"

            # Make the API call
            completion = openai.ChatCompletion.create(
                # model="gpt-4",
                model="gpt-3.5-turbo-16k",
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract and write the response
            response = completion.choices[0].message['content']
            train_file.write(response + "\n\n")
            train_file.flush()  # Flush the buffer to ensure the content is written

            # Sleep for 200 ms between API calls
            time.sleep(0.2)
