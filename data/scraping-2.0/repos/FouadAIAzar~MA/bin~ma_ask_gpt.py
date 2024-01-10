#!/usr/bin/python3

import openai
import os
import time
import psycopg2
from datetime import datetime
from config import DATABASE_PASSWORD

def print_with_typing_effect(text, delay=0.005):
    """Print text with a typing effect."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()  # Newline at the end

def store_response_to_file(response):
    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp}.txt"
    file_path = os.path.expanduser("~/MA/doc/gpt/") + file_name
    
    # Store the response to the file
    with open(file_path, 'w') as file:
        file.write(response)

    return file_name, file_path

def store_info_to_db(file_name, file_path, prompt, response):
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password=DATABASE_PASSWORD,
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    # Get file metadata
    last_modified = datetime.now()
    file_size = os.path.getsize(file_path)

    # Insert the data into the MA_GPT table
    cursor.execute(
        """
        INSERT INTO ma_gpt (file_name, file_path, last_modified, size, prompt, content)
        VALUES (%s, %s, %s, %s, %s, %s);
        """,
        (file_name, file_path, last_modified, file_size, prompt,response)
    )
    conn.commit()

    # Close the connection
    cursor.close()
    conn.close()

def ask_gpt4():
    # Fetch the OpenAI API key from environment variable
    openai_key = os.environ.get('OPENAI_KEY')
    if not openai_key:
        raise ValueError("Please set your OpenAI API key as the 'OPENAI_KEY' environment variable.")
    
    openai.api_key = openai_key
    
    # Get user input
    prompt = input("Enter your prompt for GPT-4: ")

    # Send the prompt to GPT-4
    response_obj = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides definitions and descriptions."},
            {"role": "user", "content": prompt}
        ]
    )
    response = response_obj.choices[0].message["content"]

    print("\nGPT-4 Response:")
    print_with_typing_effect(response)

    # Store response to file and get its metadata
    file_name, file_path = store_response_to_file(response)

    # Store information to the PostgreSQL table
    store_info_to_db(file_name, file_path, prompt, response)

if __name__ == "__main__":
    ask_gpt4()
