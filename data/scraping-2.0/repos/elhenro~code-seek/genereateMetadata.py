import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

def generate_description_and_tags(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Describe the following text: {text[:100]}..."},
        ]
    )
    description = response['choices'][0]['message']['content'].strip()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Give tags for the following text: {text[:100]}..."},
        ]
    )
    tags = response['choices'][0]['message']['content'].strip().split(',')

    return description, tags

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.html'):
            with open(os.path.join(directory, filename), 'r') as f:
                text = f.read()

                description, tags = generate_description_and_tags(text)

                with open(os.path.join(directory, f'{filename}-metadata.md'), 'w') as metadata_file:
                    metadata_file.write(f'## {filename}\n\n')
                    metadata_file.write(f'**Description:** {description}\n\n')
                    metadata_file.write('**Tags:**\n\n')
                    for tag in tags:
                        metadata_file.write(f'- {tag.strip()}\n')

process_directory('project/src/')