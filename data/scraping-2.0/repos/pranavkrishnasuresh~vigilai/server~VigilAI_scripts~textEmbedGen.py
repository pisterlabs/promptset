import os
from openai import OpenAI

client = OpenAI()

# Folder paths
input_folder = '/Users/krishnasuresh/Desktop/embeddedvector'
output_file = '/Users/krishnasuresh/Desktop/embeddedvector/final_vals.txt'

# Model and table details
model_name = 'text-embedding-ada-002'
table_name = 'myvectortable'

# Open the output file in append mode
with open(output_file, 'a', encoding='utf-8') as finalvals_file:

    # Traverse through each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)

            # Read the content of the text file
            with open(file_path, 'r', encoding='utf-8') as text_file:
                text_content = text_file.read()

            # Make an API request to ChatGPT for text embeddings
            def get_embedding(text, model="text-embedding-ada-002"):
                text = text.replace("\n", " ")
                return client.embeddings.create(input = [text], model=model)

            # Extract the vector embedding from the API response
            vector = get_embedding(text_content, model='text-embedding-ada-002')

            # Format the SQL INSERT statement and write it to the output file
            sql_insert = f'INSERT INTO {table_name} (text, vector) VALUES ("{text_content}", JSON_ARRAY_PACK({vector}))\n'
            finalvals_file.write(sql_insert)

print(f"Embeddings generated and saved to {output_file}")
