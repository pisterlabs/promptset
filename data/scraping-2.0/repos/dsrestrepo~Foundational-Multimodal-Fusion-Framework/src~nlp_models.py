""" Evaluate Medical Tests Classification in LLMS """
## Setup
#### Load the API key and libaries.
import os
import re

import json
import pandas as pd
import argparse
import subprocess

# Create a class to handle the GPT API
class GPT:
    # build the constructor
    def __init__(self, model='gpt-3.5-turbo', temperature=0.0, n_repetitions=1, reasoning=False, languages=['english', 'portuguese'], path='data/Portuguese.csv', max_tokens=500):
        
        import openai
        from dotenv import load_dotenv, find_dotenv
        _ = load_dotenv(find_dotenv()) # read local .env file
        openai.api_key  = os.environ['OPENAI_API_KEY']

        self.path = path
        self.model = model
        self.temperature = temperature
        self.n_repetitions = n_repetitions if n_repetitions > 0 else 1
        self.reasoning = reasoning
        self.languages = languages
        self.max_tokens = max_tokens
        
        self.delimiter = "####"
        self.responses = ['A', 'B', 'C', 'D']
        self.extra_message = ""

        if self.reasoning:
            self.output_keys = ['response', 'reasoning']
        else:
            self.output_keys = ['response']

        self.update_system_message()


    def update_system_message(self):
        """
        Update the system message based on the current configuration.
        """

        if self.reasoning:
            self.system_message = f"""
            You will be provided with medical queries in this languages: {", ".join(self.languages)}. \
            The medical query will be delimited with \
            {self.delimiter} characters.
            Each question will have {len(self.responses)} possible answer options.\
            provide the letter with the answer and a short sentence answering why the answer was selected. \
            {self.extra_message}

            Provide your output in json format with the \
            keys: {", ".join(self.output_keys)}.

            Responses: {", ".join(self.responses)}.

            """
        else:
            self.system_message = f"""
            You will be provided with medical queries in this languages: {", ".join(self.languages)}. \
            The medical query will be delimited with \
            {self.delimiter} characters.
            Each question will have {len(self.responses)} possible answer options.\
            provide the letter with the answer.
            {self.extra_message}

            Provide your output in json format with the \
            keys: {", ".join(self.output_keys)}.

            Responses: {", ".join(self.responses)}.
            
            """

    # function to change the delimiter
    def change_delimiter(self, delimiter):
        """ Change the delimiter """
        self.delimiter = delimiter        
        self.update_system_message()

    # function to change the responses
    def change_responses(self, responses):
        self.responses = responses
        self.update_system_message()
    
    def change_output_keys(self, output_keys):
        self.output_keys = output_keys
        self.update_system_message()
    
    def add_output_key(self, output_key):
        self.output_keys.append(output_key)
        self.update_system_message()

    def change_languages(self, languages):
        self.languages = languages
        self.update_system_message()
    
    def add_extra_message(self, extra_message):
        self.extra_message = extra_message
        self.update_system_message()
    
    def change_system_message(self, system_message):
        self.system_message = system_message

    def change_reasoning(self, reasoning=None):
        if type(reasoning) == bool:
            self.reasoning = reasoning
        else:
            if reasoning:
                print(f'Reasoning should be boolean. Changing reasoning from {self.reasoning} to {not(self.reasoning)}.')        
            self.reasoning = False if self.reasoning else True
        
        if self.reasoning:
            self.output_keys.append('reasoning')
            # remove duplicates
            self.output_keys = list(set(self.output_keys))
        else:
            try:
                self.output_keys.remove('reasoning')
            except:
                pass
        self.update_system_message()

    #### Template for the Questions
    def generate_question(self, question):

        user_message = f"""/
        {question}"""
        
        messages =  [  
        {'role':'system', 
        'content': self.system_message}, 
        {'role':'user', 
        'content': f"{self.delimiter}{user_message}{self.delimiter}"},  
        ] 
        
        return messages
    

    def get_embedding(self, text):
        from openai import OpenAI
        client = OpenAI()

        text = text.replace("\n", " ")

        return client.embeddings.create(input = [text], model=self.model)['data'][0]['embedding']

    def get_embedding_df(self, column, directory, file):
        df = pd.read_csv(self.path)
        df["embeddings"] = df[column].apply(lambda x: self.get_embedding(x))

        os.makedirs(directory, exist_ok=True) 
        df.to_csv(f"{directory}/{file}", index=False)

        
    #### Get the completion from the messages
    def get_completion_from_messages(self, prompt):
        
        messages = self.generate_question(prompt)

        try:        
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature, 
                max_tokens=self.max_tokens,
                request_timeout=10
            )
        except:
            response = self.get_completion_from_messages(prompt)
            return response

        response = response.choices[0].message["content"]

        # Convert the string into a JSON object
        response = json.loads(response)
    
        return response


        ### Questions from a csv file:
        df = pd.read_csv(self.path)

        ### Evaluate the model in question answering per language:
        responses = {}
        for key in self.output_keys:
            responses[key] = {}
            for language in self.languages:
                responses[key][language] = [[] for n in range(self.n_repetitions)]

        for row in range(df.shape[0]):
            print('*'*50)
            print(f'Question {row+1}: ')
            for language in self.languages:
                print(f'Language: {language}')                   
                question = df[language][row]                    
                print('Question: ')
                print(question)                        
                for n in range(self.n_repetitions): 
                    print(f'Test #{n}: ')
                    response = self.get_completion_from_messages(question)
                    print(response)
                    for key in self.output_keys:
                        # Append to the list:
                        responses[key][language][n].append(response[key])
            print('*'*50)

        ### Save the results in a csv file:
        for language in self.languages:
            if self.n_repetitions == 1:
                for key in self.output_keys:
                    df[f'{key}_{language}'] = responses[key][language][0]
            else:
                for n in range(self.n_repetitions):
                    for key in self.output_keys:
                        df[f'{key}_{language}_{n}'] = responses[key][language][n]
        if save:
            if not os.path.exists('responses'):
                os.makedirs('responses')
            if self.n_repetitions == 1:
                df.to_csv(f"responses/{self.model}_Temperature{str(self.temperature).replace('.', '_')}.csv", index=False)
            else:
                df.to_csv(f"responses/{self.model}_Temperature{str(self.temperature).replace('.', '_')}_{self.n_repetitions}Repetitions.csv", index=False)

        return df

    
    
# Create a class to handle the LLAMA 2
class LLAMA:
    # build the constructor
    def __init__(self, model='Llama-2-7b', embeddings=False, temperature=0.0, n_repetitions=1, reasoning=False, languages=['english', 'portuguese'], path='data/Portuguese.csv', max_tokens=500, verbose=False):
        
        self.embeddings = embeddings
        
        self.model = model
        model_path = self.download_hugging_face_model(model)
        
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, embedding=self.embeddings, verbose=verbose)
        
        self.path = path
        
        self.temperature = temperature
        self.n_repetitions = n_repetitions if n_repetitions > 0 else 1
        self.reasoning = reasoning
        self.languages = languages
        self.max_tokens = max_tokens
        
        
        self.delimiter = "####"
        self.responses = ['A', 'B', 'C', 'D']
        self.extra_message = ""

        if self.reasoning:
            self.output_keys = ['response', 'reasoning']
        else:
            self.output_keys = ['response']

        self.update_system_message()


    def update_system_message(self):
        """
        Update the system message based on the current configuration.
        """

        if self.reasoning:
            self.system_message = f"""
            You will be provided with medical queries in this languages: {", ".join(self.languages)}. \
            The medical query will be delimited with \
            {self.delimiter} characters.
            Each question will have {len(self.responses)} possible answer options.\
            provide the letter with the answer and a short sentence answering why the answer was selected. \
            {self.extra_message}

            Provide your output in json format with the \
            keys: {", ".join(self.output_keys)}. Make sure to always use the those keys, do not modify the keys.
            Be very careful with the resulting JSON file, make sure to add curly braces, quotes to define the strings, and commas to separate the items within the JSON.

            Responses: {", ".join(self.responses)}.
            """
        else:
            self.system_message = f"""
            You will be provided with medical queries in this languages: {", ".join(self.languages)}. \
            The medical query will be delimited with \
            {self.delimiter} characters.
            Each question will have {len(self.responses)} possible answer options.\
            {self.extra_message}

            Provide your output in json format with the \
            keys: {", ".join(self.output_keys)}. Make sure to always use the those keys, do not modify the keys.
            Be very careful with the resulting JSON file, make sure to add curly braces, quotes to define the strings, and commas to separate the items within the JSON.

            Responses: {", ".join(self.responses)}.
            """
    def download_and_rename(self, url, filename):
        """Downloads a file from the given URL and renames it to the given new file name.

        Args:
            url: The URL of the file to download.
            new_file_name: The new file name for the downloaded file.
        """

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        print(f'Downloading the weights of the model: {url} ...')
        subprocess.run(["wget", "-q", "-O", filename, url])
        print(f'Done!')
        
    def download_hugging_face_model(self, model_version='Llama-2-7b'):
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

        self.download_and_rename(MODEL_URL, model_path)

        return model_path

    # function to change the delimiter
    def change_delimiter(self, delimiter):
        """ Change the delimiter """
        self.delimiter = delimiter        
        self.update_system_message()

    # function to change the responses
    def change_responses(self, responses):
        self.responses = responses
        self.update_system_message()
    
    def change_output_keys(self, output_keys):
        self.output_keys = output_keys
        self.update_system_message()
    
    def add_output_key(self, output_key):
        self.output_keys.append(output_key)
        self.update_system_message()

    def change_languages(self, languages):
        self.languages = languages
        self.update_system_message()
    
    def add_extra_message(self, extra_message):
        self.extra_message = extra_message
        self.update_system_message()
    
    def change_system_message(self, system_message):
        self.system_message = system_message

    def change_reasoning(self, reasoning=None):
        if type(reasoning) == bool:
            self.reasoning = reasoning
        else:
            if reasoning:
                print(f'Reasoning should be boolean. Changing reasoning from {self.reasoning} to {not(self.reasoning)}.')        
            self.reasoning = False if self.reasoning else True
        
        if self.reasoning:
            self.output_keys.append('reasoning')
            # remove duplicates
            self.output_keys = list(set(self.output_keys))
        else:
            try:
                self.output_keys.remove('reasoning')
            except:
                pass
        self.update_system_message()

    #### Template for the Questions
    def generate_question(self, question):

        user_message = f"""/
        {question}"""
        
        messages =  [  
        {'role':'system', 
        'content': self.system_message}, 
        {'role':'user', 
        'content': f"{self.delimiter}{user_message}{self.delimiter}"},  
        ] 
        
        return messages

    def get_embedding(self, text):
        
        if  self.index % 5000 == 0:
            print(f'{self.index} Embeddings generated!')
            
        self.index += 1 

        text = text.replace("\n", " ")

        return self.llm.create_embedding(input = [text])['data'][0]['embedding']

    def get_embedding_df(self, column, directory, file):
        self.index = 0
        df = pd.read_csv(self.path)
        df["embeddings"] = df[column].apply(lambda x: self.get_embedding(x))

        os.makedirs(directory, exist_ok=True) 
        df.to_csv(f"{directory}/{file}", index=False)


    
    #### Get the completion from the messages
    def get_completion_from_messages(self, prompt):
        
        messages = self.generate_question(prompt)

        response = self.llm.create_chat_completion(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens)
        
        self.llm.set_cache(None)

        response = response['choices'][0]['message']["content"]        

        # Convert the string into a JSON object
        try:
            # Use regular expressions to extract JSON
            json_pattern = r'\{.*\}'  # Match everything between '{' and '}'
            match = re.search(json_pattern, response, re.DOTALL)
            response = match.group()

            # Define a regex pattern to identify unquoted string values
            pattern = r'("[^"]*":\s*)([A-Za-z_][A-Za-z0-9_]*)'
            # Use a lambda function to add quotes to unquoted string values
            response = re.sub(pattern, lambda m: f'{m.group(1)}"{m.group(2)}"', response)
            
            # Convert
            response = json.loads(response)
        except:
            print(f'Error converting respose to json: {response}')
            print('Generating new response...')
            response = self.get_completion_from_messages(prompt)
            return response
        
        if self.reasoning:
            # Iterate through the keys of the dictionary
            for key in list(response.keys()):
                if 'reas' in key.lower():
                    # Update the dictionary with the new key and its corresponding value
                    response['reasoning'] = response.pop(key)
            
        return response


    def llm_language_evaluation(self, save=True):

        ### Questions from a csv file:
        df = pd.read_csv(self.path)

        ### Evaluate the model in question answering per language:
        responses = {}
        for key in self.output_keys:
            responses[key] = {}
            for language in self.languages:
                responses[key][language] = [[] for n in range(self.n_repetitions)]

        for row in range(df.shape[0]):
            print('*'*50)
            print(f'Question {row+1}: ')
            for language in self.languages:
                print(f'Language: {language}')                   
                question = df[language][row]                    
                print('Question: ')
                print(question)                        
                for n in range(self.n_repetitions): 
                    print(f'Test #{n}: ')
                    response = self.get_completion_from_messages(question)
                    print(response)
                    for key in self.output_keys:
                        # Append to the list:
                        responses[key][language][n].append(response[key])
            print('*'*50)
        
        

        ### Save the results in a csv file:
        for language in self.languages:
            if self.n_repetitions == 1:
                for key in self.output_keys:
                    df[f'{key}_{language}'] = responses[key][language][0]
            else:
                for n in range(self.n_repetitions):
                    for key in self.output_keys:
                        df[f'{key}_{language}_{n}'] = responses[key][language][n]
        if save:
            if not os.path.exists('responses'):
                os.makedirs('responses')
            if self.n_repetitions == 1:
                df.to_csv(f"responses/{self.model}_Temperature{str(self.temperature).replace('.', '_')}.csv", index=False)
            else:
                df.to_csv(f"responses/{self.model}_Temperature{str(self.temperature).replace('.', '_')}_{self.n_repetitions}Repetitions.csv", index=False)

        return df