import logging
import re
from openai import OpenAI
import pprint
import tenacity
import tiktoken
from autoPDFtagger.config import config
api_key = config['OPENAI-API']['API-Key']
LANGUAGE = config['DEFAULT']['language']

class AIAgent:
    def __init__(self): 
        self.log_file = "" # set a filename to enable logging of Communication to seperate file

    def send_request(self, user_message):
        pass

    # Try to repair a corrupt JSON
    def clean_json(self, json_text):
        # remove additional commas
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)

        # Looking for a JSON-Objekt
        match = re.search(r'\{.*\}', json_text, re.DOTALL)
        if match:
            return match.group(0)
        return None
    
    def write_to_log_file(self, text):
        if self.log_file:
            try:
                with open(self.log_file, 'a') as log:
                    log.write(text)
            except Exception as e:
                logging.error("Error logging AIAgent-Logfile: {}".format(e))
    


OpenAI_model_pricelist = { # $ per 1 k token, [input, output, token_limit]
    "gpt-3.5-turbo-1106": [0.001, 0.002, 16385],
    "gpt-4-1106-preview": [0.01, 0.03, 4096], # Max  = 128000, reduced for cost reasons
    "gpt-4-vision-preview": [0.01, 0.03, 4096]
}

class AIAgent_OpenAI(AIAgent):
    def __init__(self, 
                 model="gpt-3.5-turbo-1106", 
                 system_message="You are a helpful assistant"):
        super().__init__()
        
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.set_model(model)
        self.messages = []
        self.add_message(system_message, role="system")

        # Cost-Control
        self.max_tokens=4096
        self.cost = 0


    def add_message(self, content, role="user"):
        self.messages.append({"role": role, "content": content})

    @tenacity.retry(
            wait=tenacity.wait_random_exponential(min=5, max=60), 
            stop=tenacity.stop_after_attempt(6))
    def send_request(self,
                    temperature=0.7,
                    response_format="text" # Alt: "object-json"
                    ):
        logging.debug("Trying to send API-Request")
        # Temporäres Ändern des Logging-Levels
        original_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)  # Ändere das Logging-Level auf ERROR

        try:
            encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')


            if response_format:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    response_format={"type": response_format},
                    temperature=temperature,
                    max_tokens=self.max_tokens
                )          
            else:  
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens
                )   

            # Logging Data in seperate file if log_file is set
            self.write_to_log_file(
                "API-REQUEST:\n" 
                + pprint.pformat(self.messages) 
                + "\n\nAPI-ANSWER:\n" 
                + pprint.pformat(response))

            self.cost += self.get_costs(response.usage.prompt_tokens, response.usage.completion_tokens)

            
            logging.getLogger().setLevel(original_level)

            return self.clean_json(response.choices[0].message.content)
            
        except Exception as e: 
            logging.error(e)
            # Restore original logging level
            logging.getLogger().setLevel(original_level)
            raise e

    def get_costs(self, token_input, token_output):
        if self.model in OpenAI_model_pricelist:
            cost_per_token_input, cost_per_token_output, limit = OpenAI_model_pricelist[self.model]
            total_cost_input = token_input * cost_per_token_input / 1000
            total_cost_output = token_output * cost_per_token_output / 1000
            return total_cost_input + total_cost_output
        else:
            raise ValueError("Model '" + self.model + "' not found in the price list.")

    def set_model(self, model):
        if model in OpenAI_model_pricelist:
            self.model = model
        else:
            raise ValueError("Model '" + model + "' not available.")
    
