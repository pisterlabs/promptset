""" Evaluate Medical Tests Classification in LLMS """
## Setup
#### Load the API key and libaries.
import json
import pandas as pd
import os
import openai
from dotenv import load_dotenv, find_dotenv
import argparse
import re
import subprocess


### Download LLAMA model:
def download_and_rename(url, filename):
    """Downloads a file from the given URL and renames it to the given new file name.

    Args:
        url: The URL of the file to download.
        new_file_name: The new file name for the downloaded file.
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print(f'Downloading the weights of the model: {url} ...')
    subprocess.run(["wget", "-q", "-O", filename, url])
    print(f'Done!')

def download_hugging_face_model(model_version='Llama-2-7b'):
    if model_version not in ['Llama-2-7b', 'Llama-2-13b', 'Llama-2-70b']:
        raise ValueError("Options for Llama model should be 7b, 13b or 70b")

    MODEL_URL = {
        'Llama-2-7b': 'https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q8_0.gguf', 
        'Llama-2-13b': 'https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q8_0.gguf', 
        'Llama-2-70b': 'https://huggingface.co/TheBloke/Llama-2-70B-chat-GGUF/resolve/main/llama-2-70b-chat.Q5_0.gguf'
    }

    MODEL_URL = MODEL_URL[model_version]

    model_path = f'Models/{model_version}.gguf'

    if os.path.exists(model_path):
        confirmation = input(f"The model file '{model_path}' already exists. Do you want to overwrite it? (yes/no): ").strip().lower()
        if confirmation != 'yes':
            print("Model installation aborted.")
            return model_path

    download_and_rename(MODEL_URL, model_path)

    return model_path

### Model GPT:
def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=500):

    try:        
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, 
            max_tokens=max_tokens,
            request_timeout=10
        )
    except:
        response = get_completion_from_messages(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        return response

    return response.choices[0].message["content"]

def get_completion_from_messages_hf(messages, 
                                    model):
    
    response = model(messages)[0]['generated_text'].replace(messages, '')
    
    return {'response': response}

#### Model Llama 2
def get_completion_from_messages_llama(messages, 
                                 model, 
                                 temperature=0, 
                                 max_tokens=500, 
                                 reasoning=False):
    # Get the response:
    response = model.create_chat_completion(
        messages,
        temperature=temperature,
        max_tokens=max_tokens)
    
    model.set_cache(None)
    
    response = response['choices'][0]['message']["content"]        

    # Convert the string into a JSON object
    # Due to some problems with Llama 2 generating JSON formats, the output requires more preprocessing than GPT.
    try:
        # Use regular expressions to extract JSON
        json_pattern = r'\{.*\}'  # Match everything between '{' and '}'
        match = re.search(json_pattern, response, re.DOTALL)
        response = match.group()
        

        # Define a regex pattern to identify unquoted string values
        pattern = r'("[^"]*":\s*)([A-Za-z_][A-Za-z0-9_]*)'
        # Use a lambda function to add quotes to unquoted string values
        response = re.sub(pattern, lambda m: f'{m.group(1)}"{m.group(2)}"', response)
        
        
        if not reasoning:
            # Convert from {'response': 'A' ) some text without quotes} to {'response': 'A'}
            # Use regular expression to extract the letter and surrounding characters
            match = re.search(r'"response": "([A-Da-d][^\"]*)"', response)

            if match:
                response = f'{{{match.group(0)}}}'

        # Convert
        response = json.loads(response)
        
        resp = response['response']
        
    except:
        print(f'Error converting respose to json: {response}')
        print('Generating new response...')
        response = get_completion_from_messages_llama(messages, model=model, temperature=temperature, max_tokens=max_tokens, reasoning=reasoning)
        return response

    if reasoning:
        # Iterate through the keys of the dictionary
        for key in list(response.keys()):
            if 'reas' in key.lower():
                # Update the dictionary with the new key and its corresponding value
                response['reasoning'] = response.pop(key)

    return response

#### Template for the Questions
def generate_question(question, LANGUAGES, REASONING, Responses=['A', 'B', 'C', 'D']):
    
    delimiter = "####"

    languages_text = ", ".join(LANGUAGES)

    if REASONING:
        system_message = f"""
        You will be provided with medical queries in this languages: {languages_text}. \
        The medical query will be delimited with \
        {delimiter} characters.
        Each question will have {len(Responses)} possible answer options.\
        provide the letter with the answer and a short sentence answering why the answer was selected \

        Provide your output in json format with the \
        keys: response and reasoning.

        Responses: {", ".join(Responses)}.

        """
    else:
        system_message = f"""
        You will be provided with medical queries in this languages: {languages_text}. \
        The medical query will be delimited with \
        {delimiter} characters.
        Each question will have {len(Responses)} possible answer options.\
        provide the letter with the answer.

        Provide your output in json format with the \
        key: response.

        Responses: {", ".join(Responses)}.

        """

    user_message = f"""/
    {question}"""
    
    messages =  [  
    {'role':'system', 
     'content': system_message},    
    {'role':'user', 
     'content': f"{delimiter}{user_message}{delimiter}"},  
    ] 
    
    return messages

def generate_template_text_generation(question, LANGUAGES, Responses=['A', 'B', 'C', 'D']):
    
    delimiter = "####"

    languages_text = ", ".join(LANGUAGES)

    messages = f"""You will be provided with medical queries in this languages: {languages_text}. \
    The medical query will be delimited with {delimiter} characters.
    Each question will have {len(Responses)} possible answer options.Provide just the letter with the answer.

    Responses: {", ".join(Responses)}.

    Question:
    {delimiter}{question}{delimiter}

    The response is: """

    return messages

#### Template for the Questions
def generate_question_llama(question, LANGUAGES, REASONING, Responses=['A', 'B', 'C', 'D']):
    
    delimiter = "####"
    
    out_template = ""
    
    if REASONING:
        output_keys = ['response', 'reasoning']
    else:
        output_keys = ['response']
        
    for response in Responses:
        response_dict = {key: f'something describing {key}' for key in output_keys}
        response_dict[output_keys[0]] = response
        response_str = ', '.join([f"'{k}': '{v}'" for k, v in response_dict.items()])
        out_template += f"If response is {response}: {{ {response_str} }}\n"

    languages_text = ", ".join(LANGUAGES)

    if REASONING:
        system_message = f"""
You will be provided with medical queries in this languages: {languages_text}. \
The medical query will be delimited with \
{delimiter} characters.
Each question will have {len(Responses)} possible answer options.\
provide just the letter with the answer and a short sentence answering why the answer was selected.

Provide your output in json format with the \
keys: response and reasoning. Make sure to always use the those keys, do not modify the keys. 

Response option: {", ".join(Responses)}.

Always use the JSON format.

The output shoulb be: {{"response": "Response option", "", ""}}
        """
    else:
        system_message = f"""
You will be provided with medical queries in this languages: {languages_text}. \
The medical query will be delimited with \
{delimiter} characters.
Each question will have {len(Responses)} possible answer options.\
provide just the letter with the answer.

Provide your output in json format with the \
key: response. Make sure to always use the that key, do not modify the key. 

Response option: {", ".join(Responses)}.

Always use the JSON format.

The output shoulb be: {{"response": "Response option"}}
        """

    user_message = f"""/
    {question}"""
    
    messages =  [  
    {'role':'system', 
     'content': system_message},    
    {'role':'user', 
     'content': f"{delimiter}{user_message}{delimiter}"},  
    ] 
    
    return messages


def llm_language_evaluation(path='data/Portuguese.csv', model='gpt-3.5-turbo', temperature=0.0, n_repetitions=1, reasoning=False, languages=['english', 'portuguese']):
    
    # Load API key if GPT, or Model if LLAMA
    if 'gpt' in model:
        _ = load_dotenv(find_dotenv()) # read local .env file
        openai.api_key  = os.environ['OPENAI_API_KEY']
    elif 'Llama-2' in model:                
        model_path = download_hugging_face_model(model_version=model)
        from llama_cpp import Llama
        llm = Llama(model_path=model_path)
    elif 'bloom':
        from transformers import pipeline
        llm = pipeline(
            task="text-generation", 
            model=model,
            model_kwargs={"temperature": temperature, "max_length": 5})
        reasoning = False
    else:
        print('Model should be a GPT or Llama-2 model')
        return 0
    
    #### Load the Constants
    PATH = path # 'data/Portuguese.csv'
    MODEL = model # "gpt-3.5-turbo"
    TEMPERATURE = temperature # 0.0
    N_REPETITIONS = n_repetitions # 1
    REASONING = reasoning # False
    LANGUAGES = languages # ['english', 'portuguese']

    if N_REPETITIONS <= 0 or (N_REPETITIONS != int(N_REPETITIONS)):
        print(f'N_REPETITIONS should be a positive integer, not {N_REPETITIONS}')
        print('N_REPETITIONS will be set to 1')
        N_REPETITIONS = 1


    ### Questions from a csv file:
    df = pd.read_csv(PATH)

    ### Evaluate the model in question answering per language:
    responses = {}
    reasoning = {}
    for language in LANGUAGES:
        responses[language] = [[] for n in range(N_REPETITIONS)]

        if REASONING:
            reasoning[language] = [[] for n in range(N_REPETITIONS)]

    for row in range(df.shape[0]):
        print('*'*50)
        print(f'Question {row+1}: ')
        for language in LANGUAGES:
            print(f'Language: {language}')
            question = df[language][row]
            print('Question: ')
            print(question)
            if 'gpt' in model: 
                messages = generate_question(question, LANGUAGES, REASONING)
            elif 'Llama-2' in model:
                messages = generate_question_llama(question, LANGUAGES, REASONING)
            elif 'bloom' in model:
                messages = generate_template_text_generation(question, LANGUAGES)
            else:
                print('Model should be a GPT or Llama-2 model')
                return 0 

            for n in range(N_REPETITIONS): 
                print(f'Test #{n}: ')
                
                if 'gpt' in model: 
                    response = get_completion_from_messages(messages, MODEL, TEMPERATURE)
                    # Convert the string into a JSON object
                    response = json.loads(response)
                elif 'Llama-2' in model:
                    response = get_completion_from_messages_llama(messages, llm, TEMPERATURE, reasoning=REASONING)
                elif 'bloom' in model:
                    response = get_completion_from_messages_hf(messages, llm)
                else:
                    print('Model should be a GPT or Llama-2 model')
                    return 0
                    
                print(response)
            
                # Append to the list:
                responses[language][n].append(response['response'])
                if REASONING:
                    reasoning[language][n].append(response['reasoning'])
                
        print('*'*50)

    ### Save the results in a csv file:
    for language in LANGUAGES:
        if N_REPETITIONS == 1:
            df[f'responses_{language}'] = responses[language][0]
            if REASONING:
                df[f'reasoning_{language}'] = reasoning[language][0]
                
        for n in range(N_REPETITIONS):
            df[f'responses_{language}_{n}'] = responses[language][n]
            if REASONING:
                df[f'reasoning_{language}_{n}'] = reasoning[language][n]

    if not os.path.exists('responses'):
        os.makedirs('responses')
    if N_REPETITIONS == 1:
        df.to_csv(f"responses/{MODEL}_Temperature{str(TEMPERATURE).replace('.', '_')}.csv", index=False)
    else:
        df.to_csv(f"responses/{MODEL}_Temperature{str(TEMPERATURE).replace('.', '_')}_{N_REPETITIONS}Repetitions.csv", index=False)

def main():
    # Add argparse code to handle command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate Medical Tests Classification in LLMS")
    parser.add_argument("--csv_file", default="data/Portuguese.csv", help="Path to the CSV file with the questions")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM to use e.g: gpt-3.5-turbo, gpt-4, Llama-2-7b, Llama-2-13b, or Llama-2-70b")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature parameter of the model between 0 and 1. Used to modifiy the model's creativity. 0 is deterministic and 1 is the most creative")
    parser.add_argument("--n_repetitions", type=int, default=1, help="Number of repetitions to run each experiment. Used to measure the model's hallucinations")
    parser.add_argument("--reasoning", action="store_true", default=False, help="Enable reasoning mode. If set to True, the model will be asked to provide a reasoning for its answer. If set to True the model uses more tokens")
    parser.add_argument("--languages", nargs='+', default=['english', 'portuguese'], help="List of languages")
    args = parser.parse_args()


    PATH = args.csv_file
    MODEL = args.model
    TEMPERATURE = args.temperature
    N_REPETITIONS = args.n_repetitions
    REASONING = args.reasoning
    LANGUAGES = args.languages

    llm_language_evaluation(path=PATH, model=MODEL, temperature=TEMPERATURE, n_repetitions=N_REPETITIONS, reasoning=REASONING, languages=LANGUAGES)


if __name__ == "__main__":
    main()