import openai
import os


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Set OpenAI API key
openai.api_key = open_file('openaiapikey.txt')
#openai.api_key = "YOUR_API_KEY"

# Set the model to use
model_engine = "text-davinci-003"

# Set the temperature and token count for the generated text
temperature = 0
token_count = 1450

# Read the prompt file and store the contents in a variable
with open("prompt_JSONLD_citation_nodes.txt", "r", encoding='utf-8') as f:
    prompt = f.read()

# Iterate through the .txt files in the "chunks_txt" folder
for file in os.listdir("chunks_txt"):
    if file.endswith(".txt"):
        # Read the contents of the file
        with open(os.path.join("chunks_txt", file), "r", encoding='utf-8') as f:
            chunk = f.read()

        # Replace the placeholder with the chunk
        prompt_with_chunk = prompt.replace("<<CHUNK>>", chunk)

        # Use the OpenAI API to generate text based on the modified prompt
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt_with_chunk,
            temperature=temperature,
            max_tokens=token_count,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )


        # Check if the "kg_json" folder exists and create it if it doesn't
        if not os.path.exists("kg_json"):
            os.makedirs("kg_json")
        
        # Print the generated text
        text = response['choices'][0]['text'].strip()
        print('\n\n\n=========================================\n\n\n', text)
        # Save the generated text as a .json file in the "kg_json" folder
        with open(os.path.join("kg_json", file.replace(".txt", ".json")), "w", encoding='utf-8') as f:
            f.write(text)
