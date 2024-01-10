import os
import json
import random
import argparse
import datetime

import backoff
import openai
import time
import tiktoken


from openai.error import RateLimitError, APIError, ServiceUnavailableError, APIConnectionError

from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", "")
assert openai_api_key, "OPENAI_API_KEY environment variable is missing from .env"

support_models = ['gpt-3.5-turbo', 'gpt-4']

model2max_context = {
    "gpt-4": 7900,
    "gpt-3.5-turbo": 3900,
    "text-davinci-003": 4096,
    "text-davinci-002": 4096,
}

results = {
    "start": 0,
    "number": 0,
    "random": False,
    "temperature": 0.0,
    "top_p": 0.0,
    "no_correct": 0,
    "unknow_answers": [],
    "outcomes": [{
        "ground_truth": "",
        "gpt_interaction": {
            "api_call": {
                "prompt": {},
                "response": {}
            }
        }
    }]
}

class OutOfQuotaException(Exception):
    "Raised when the key exceeded the current quota"
    def __init__(self, key, cause=None):
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

class AccessTerminatedException(Exception):
    "Raised when the key has been terminated"
    def __init__(self, key, cause=None):
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

@backoff.on_exception(backoff.expo, (RateLimitError, APIError, ServiceUnavailableError, APIConnectionError), max_tries=20)
def query(messages: "list[dict]", max_tokens: int, api_key: str, temperature: float, top_p: float, sleep_time: float, model_name: str, ground_truth: str) -> str:
    """make a query

    Args:
        messages (list[dict]): chat history in turbo format
        max_tokens (int): max token in api call
        api_key (str): openai api key
        temperature (float): sampling temperature

    Raises:
        OutOfQuotaException: the apikey has out of quota
        AccessTerminatedException: the apikey has been ban

    Returns:
        str: the return msg
    """
    time.sleep(sleep_time)
    assert model_name in support_models, f"Not support {model_name}. Choices: {support_models}"
    try:
        if model_name in support_models:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                api_key=api_key,
            )
            
            result = {
                    "ground_truth": ground_truth,
                    "gpt_interaction": {
                        "api_call": {
                            "prompt": message,
                            "response": response
                        }
                    }
            }

            results['outcomes'].append(result)

            gen = response['choices'][0]['message']['content']
            #print(gen, response)
        return gen

    except RateLimitError as e:
        if "You exceeded your current quota, please check your plan and billing details" in e.user_message:
            raise OutOfQuotaException(api_key)
        elif "Your access was terminated due to violation of our policies" in e.user_message:
            raise AccessTerminatedException(api_key)
        else:
            raise e

def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input-file", type=str, required=True, help="Input file path")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output file dir")
    parser.add_argument("-m", "--model-name", type=str, default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("-tp", "--top-p", type=float, default=0, help="Sampling Top P")
    parser.add_argument("-s", "--start", type=float, default=0, help="Data index to start from")
    parser.add_argument("-n", "--number", type=float, default=100, help="Number of data elements to iterate through")
    parser.add_argument("-pro", "--proteus", type=bool, default=False, help="Use Proteus prompt")
    parser.add_argument("-pr", "--pre", type=bool, default=False, help="Use pre prompts")
    parser.add_argument("-po", "--post", type=bool, default=False, help="Use post prompts")
    parser.add_argument("-r", "--random", type=bool, default=False, help="Use random selection from question set based on start and number")

    return parser.parse_args()

def extract_answer(rawanswer):
    extracted_answer = {
        "answer": ""
    }

    if "{'answer':" in rawanswer:
        start_index = rawanswer.index("{'answer':")
        extracted_string = rawanswer[start_index:]
        end_index = extracted_string.index("}")
        extracted_answer = extracted_string[:end_index+1]


    if '{"answer":' in rawanswer:
        start_index = rawanswer.index('{"answer":')
        extracted_string = rawanswer[start_index:]
        end_index = extracted_string.index("}")
        extracted_answer = extracted_string[:end_index+1]
    
    if '{"\nanswer":' in rawanswer:
        start_index = rawanswer.index('{"answer":')
        extracted_string = rawanswer[start_index:]
        end_index = extracted_string.index("\n}")
        extracted_answer = extracted_string[:end_index+1] 

    return extracted_answer

if __name__ == "__main__":

    current_script_path = os.path.abspath(__file__)
    MAD_path = current_script_path.rsplit("/", 1)[0]

    args = parse_args()

    #api_key = args.api_key
    model_name = args.model_name
    temperature = args.temperature
    top_p = args.top_p
    input_file = args.input_file
    data = []
    start = args.start
    number = args.number
    proteus = args.proteus
    proteus_prompt = ""
    pre = args.pre
    pre_prompt = ""
    post = args.post
    post_prompt = ""
    prompt = ""

    if input_file is not None:
        with open(input_file) as f:
            data = [json.loads(line) for line in f]
            if start is not None and number is not None:
                data = data[int(start):int(start + number)]
    j = 0
    for i in data:
        question = i['question']
        answer_raw = i['answer']
        answer = ""

        answer_index = answer_raw.index('#### ')
        answer = answer_raw[answer_index+5:]
        print(j, " Ground truth answer: ", answer)


        if proteus:
            proteus_prompt = json.load(open(f"{MAD_path}/code/utils/proteus.json", "r"))['prompt'] 
        if pre: 
            pre_prompt = json.load(open(f"{MAD_path}/code/utils/preprompt.json", "r"))['prompt']        
        if post: 
            post_prompt = json.load(open(f"{MAD_path}/code/utils/postprompt.json", "r"))['prompt']         
        
        prompt = pre_prompt + question + post_prompt

        message = [{"role": "system", "content": f"{proteus_prompt}"}]
        message.append({"role": "user", "content": f"{prompt}"})

        num_context_token = sum([num_tokens_from_string(m["content"], model_name) for m in message])
        max_token = model2max_context[model_name] - num_context_token
        response = query(message, max_token, api_key=openai_api_key, temperature=temperature, top_p=top_p, sleep_time=0, model_name=model_name, ground_truth=answer)
        response_answer = extract_answer(response)
        try:
            if str(answer) in str(response_answer):
                results['no_correct'] += 1
            else:
                results["unknow_answers"].append(j)
        except Exception as e:
            results["unknow_answers"].append(j)
            print(e)
        print(response)

        j += 1

    results['start'] = start
    results['number'] = number
    results["temperature"] = temperature
    results["top_p"] = top_p
    
    presentDate = datetime.datetime.now()
    unix_timestamp = datetime.datetime.timestamp(presentDate)*1000
    file_path = f"{MAD_path}/data/output/start_{start}_number_{number}_proteus_{proteus}_pre_{pre}_post_{post}_model_{model_name}_{unix_timestamp}.json"
    file = open(file_path, 'w')
    file.write(json.dumps(results))
    file.close()
    print(results)

