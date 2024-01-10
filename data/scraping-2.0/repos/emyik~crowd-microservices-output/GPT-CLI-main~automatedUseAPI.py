import openai
import json
from decouple import config

API_KEY = config("OPENAI_KEY")
openai.api_key = API_KEY

# Terminal Color Defenitions:
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
NORMAL = "\033[0m"

def textCompletion(prompt):
    model, temp = "text-davinci-003", 0.5 # CAN CHANGE
    response = openai.Completion.create(
        model = model,
        prompt = prompt,
        temperature = temp,
        max_tokens = 2000
    )
    response = json.loads(str(response))
    content = response['choices'][0]['text']

    with open("service/shopping_service.js", "a") as file: # CHANGE
        file.write(content)
    return content

def extract_function_info(file_path):
    functions = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_function = None
        current_description = None
        for line in lines:
            if line.strip() == '':
                if current_function is not None and current_description is not None:
                    functions.append({'name': current_function, 'description': current_description})
                    current_function = None
                    current_description = None
            elif current_function is None:
                current_function = line.strip()
            elif current_description is None:
                current_description = line.strip()
            else:
                current_description += "\n" + line.strip()
    functions.append({'name': current_function, 'description': current_description})
    return functions

def main():
    file_path = 'GPT-CLI-main/shopping_functionDescriptions.txt'
    # get function names and descriptions
    functions = extract_function_info(file_path)
    for f in functions:        
        # fill in prompt template with placeholders
        with open('GPT-CLI-main/prompt_template2.txt', 'r') as file: template = file.read()
        formatted_text = template.format(function=f['name'], description=f['description'])
        textCompletion(formatted_text)    
    
    # add the rest of the necessary code to microservice file
    # with open("GPT-CLI-main/microservice_template.txt", 'r') as source: text = source.read()
    # with open("service/todo_microservice2.js", 'a') as destination: # CHANGE
    #     destination.write(text)
    #     destination.close()


if __name__ == "__main__":
    main()