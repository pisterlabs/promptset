import cohere 
import os

api_key = 'y7Pemp9bBQAX1DUwtP8bFtKssS4qSudAlzhQh87S'
co = cohere.Client(api_key)

prompt_path = 'prompt\prompt_5_promissorporra - sem resposta.txt'
# Reading the entire file content at once
with open(prompt_path, 'r') as file:
    prompt = file.read()

model = 'command-nightly'
# model = 'e0e24dd3-2818-4af4-b847-57a0f244e277-ft'
# model = 'base'
response = co.generate(  
    model=model,  
    prompt = prompt,
    # max_tokens=200, # This parameter is optional. 
    temperature=0.7,
    max_tokens=200)

intro_paragraph = response.generations[0].text

# response = co.generate(
#   model='f2e29a92-b7c1-44b0-8344-610a442ac4d2-ft',
#   prompt=prompt)

def write_to_file(response_text, base_filename="output_response.txt"):
    # Verifica se o arquivo já existe
    if os.path.exists(base_filename):
        # Se o arquivo existir, encontre um novo nome de arquivo
        index = 1
        while os.path.exists(f"{index}_{base_filename}"):
            index += 1
        filename = f"{index}_{base_filename}"
    else:
        filename = base_filename
    
    # Escreve o texto no arquivo
    with open(filename, 'w') as file:
        file.write(response_text)
    print(f"Response written to: {filename}")


response = response.generations[0].text
print('Prediction: {}'.format(response))
write_to_file(response, base_filename="output_response.txt")
write_to_file(prompt, base_filename="prompt.txt")