import sys
import os

import replicate
import openai

import requests
import json
import re

API_OPENAPI = "OpenAI"
API_REPLICATE = "Replicate"
API_OLLAMA = "Ollama"

LLM_API_IDS = [
     API_OPENAPI, API_REPLICATE, API_OLLAMA
]

LLM_API_ENDPOINT_DEFAULTS = {
    API_OLLAMA: 'http://127.0.0.1:11434/api/generate'
    # OpenAPI and Replicate endpoints are set by libraries
}

LLM_BY_ID = {
    API_OLLAMA + '/Orca2-13B': 'orca2:13b',
    API_OLLAMA + '/Neural-Chat': 'neural-chat',
    API_OLLAMA + '/OpenChat': 'openchat',
    API_OLLAMA + '/Mistral': 'mistral',
    API_OLLAMA + '/OpenHermes-2.5-Mistral': 'openhermes2.5-mistral',
    API_OLLAMA + '/Llama2': 'llama2',
    API_OLLAMA + '/Llama2-70B-Chat': 'llama2:70b-chat',
    API_OLLAMA + '/Stable-Beluga': 'stable-beluga',
    API_OLLAMA + '/Stable-Beluga-70B': 'stable-beluga:70b',
    API_OLLAMA + '/Vicuna': 'vicuna',
    API_OLLAMA + '/Vicuna-33B': 'vicuna:33b',
    API_OLLAMA + '/Wizard-Math': 'wizard-math',
    API_OLLAMA + '/Wizard-Math-70B': 'wizard-math:70b',
    API_OLLAMA + '/Yi': 'yi',
    API_OLLAMA + '/Yi-34B-Chat': 'yi:34b-chat',
    API_OLLAMA + '/Yi-34B-Chat-Q4_K_M': 'yi:34b-chat-q4_K_M',
    API_OLLAMA + '/Zephyr': 'zephyr',
    API_OPENAPI + '/gpt-4': 'gpt-4',
    API_OPENAPI + '/gpt-3.5-turbo': 'gpt-3.5-turbo',
    API_OPENAPI + '/gpt-3.5-turbo-16k': 'gpt-3.5-turbo-16k',
    API_OPENAPI + '/gpt-3.5-turbo-instruct': 'gpt-3.5-turbo-instruct',
    API_REPLICATE + '/Llama2-70B-Chat': 'replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf',
    API_REPLICATE + '/Llama2-70B': 'replicate/llama70b-v2-chat:e951f18578850b652510200860fc4ea62b3b16fac280f83ff32282f87bbd2e48', 
    API_REPLICATE + '/Llama2-13B': 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
    API_REPLICATE + '/Llama2-7B': 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea',
    API_REPLICATE + '/Mistral-7B-Instruct': 'mistralai/mistral-7b-instruct-v0.1:83b6a56e7c828e667f21fd596c338fd4f0039b46bcfa18d973e8e70e455fda70',
#    API_REPLICATE + '/yi-34b': 'nateraw/yi-34b:d83ccf090ccd5c7fe507ca302a558a850468293385d02bb807ee2753d802dd85',
#    API_REPLICATE + '/yi-34b-200k': 'nateraw/yi-34b-200k:9060f47173629dc2a77c067b7c4a5d260dc1e32e303a99298994d0ab8e903118',
#    API_REPLICATE + '/VLLM-Engine-Llama-7B': 'moinnadeem/vllm-engine-llama-7b:04bca4ff7a051e666f17a2c62a35d834e0e6fbfbd22ee212c7ba579d243450e1',
#    API_REPLICATE + '/Defog-SQLCoder-Q8-15B': 'gregwdata/defog-sqlcoder-q8:0a9abc0d143072fd5d8920ad90b8fbaafaf16b10ffdad24bd897b5bffacfce0b',
    API_REPLICATE + '/CodeLlama-34B-Instruct-GGUF': 'andreasjansson/codellama-34b-instruct-gguf:f1091fa795c142a018268b193c9eea729e0a3f4d55d723df0b69f17b863bf5ea',
    API_REPLICATE + '/WizardCoder-Python-34B': 'andreasjansson/wizardcoder-python-34b-v1-gguf:67eed332a5389263b8ede41be3ee7dc119fa984e2bde287814c4abed19a45e54',
    API_REPLICATE + '/CodeLlama-13B-Instruct': 'meta/codellama-13b-instruct:be553392065353425e0f0193d2a896d6a5ff201549f5d7cd9180c8dfdeac39ed',
    API_REPLICATE + '/Falcon-40B-Instruct': 'joehoover/falcon-40b-instruct:7eb0f4b1ff770ab4f68c3a309dd4984469749b7323a3d47fd2d5e09d58836d3c'
}

PARAMETER_DEFAULTS = {
    API_OPENAPI: {
        "temperature": 0.2,
        "top_p": 0.9,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    },
    API_REPLICATE: {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_new_tokens": 4096,
        "min_new_tokens": -1,
        "top_k": 50
        #"stop_sequences": ""
    },
    API_OLLAMA: {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 50
        #"num_keep": 5,
        #"seed": 42,
        #"num_predict": 100,
        #"tfs_z": 0.5,
        #"typical_p": 0.7,
        #"repeat_last_n": 33,
        #"repeat_penalty": 1.2,
        #"presence_penalty": 1.5,
        #"frequency_penalty": 1.0,
        #"mirostat": 1,
        #"mirostat_tau": 0.8,
        #"mirostat_eta": 0.6,
        #"penalize_newline": true,
        #"stop": ["\n", "user:"],
        #"numa": false,
        #"num_ctx": 4,
        #"num_batch": 2,
        #"num_gqa": 1,
        #"num_gpu": 1,
        #"main_gpu": 0,
        #"low_vram": false,
        #"f16_kv": true,
        #"logits_all": false,
        #"vocab_only": false,
        #"use_mmap": true,
        #"use_mlock": false,
        #"embedding_only": false,
        #"rope_frequency_base": 1.1,
        #"rope_frequency_scale": 0.8,
        #"num_thread": 8
    }
}

CTX_TEMPLATE = "You are a helpful assistant. You do not respond as 'user' or pretend to be 'user'. You only respond once as 'assistant'."

ROLE = "role"
ROLE_AS = "assistant"
ROLE_US = "user"

MSG = "message"
MSG_FORMAT = "format"
MSG_FORMAT_TXT = "text"

class LLMApiClient:
    """Requests running a LLM through an API"""

    def __init__(self):
        print("Load LLM API client ...")


    def initialize_llm(self, selected_llm, selected_llm_api_id, llm_parameters, api_key, api_endpoint):
        """Sets LLM and parameters with API keys"""

        print("Initialize LLM API client ...")

        # Store API keys in environment variables
        if selected_llm.startswith(API_OPENAPI):
            os.environ["OPENAI_API_KEY"] = api_key
        elif selected_llm.startswith(API_REPLICATE):
            os.environ["REPLICATE_API_TOKEN"] = api_key
        
        self.api_key = api_key

        self.selected_llm = selected_llm
        self.selected_llm_api_id = selected_llm_api_id

        self.llm_parameters = llm_parameters

        # Set API endpoint
        if selected_llm.startswith(API_OLLAMA) and api_endpoint:
            self.api_endpoint = api_endpoint
        elif selected_llm.startswith(API_OLLAMA):
            self.api_endpoint = LLM_API_ENDPOINT_DEFAULTS[API_OLLAMA]

        # Context returned by some APIs
        self.llm_returned_context = []

        # LLM response is streamed and not finished
        self.llm_response_ongoing = False
        self.llm_response_buffer = []

    def run_llm_replicate(self, llm_id, dialogue):
        """Run LLM with the replicate API"""

        # https://replicate.com/meta/llama-2-70b-chat
        api_parameters = { "prompt": f"{dialogue} {ROLE_AS}: " } #"repetition_penalty": 1, "stop_sequences": ""

        # add parameters
        for p in self.llm_parameters.keys():
            api_parameters[p] = self.llm_parameters[p]

        #print(self.llm_parameters)
        #print(api_parameters)

        response = replicate.run(llm_id, input=api_parameters)
        return response

    def run_llm_openai(self, llm_id, dialogue):
        """Run LLM with the OpenAI API"""

        # https://platform.openai.com/docs/api-reference/chat/create

        api_parameters = {
            "model": llm_id,
            "messages": dialogue,
            "stream": True
        }

        # add parameters
        for p in self.llm_parameters.keys():
            api_parameters[p] = self.llm_parameters[p]

        #print(self.llm_parameters)
        #print(api_parameters)

        openai.api_key = self.api_key

        result = openai.ChatCompletion.create(**api_parameters)
        return result

    def run_llm_rest_ollama(self, llm_id, dialogue):
        """Run LLM with the given Rest API"""

        # https://github.com/jmorganca/ollama/blob/main/docs/api.md
        # https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
        api_parameters = {
            'model': llm_id, 
            'prompt': dialogue,
            'stream': True,
            'options': {}
        }

        if len(self.llm_returned_context) > 0:
            api_parameters['context'] = self.llm_returned_context

        # add parameters
        for p in self.llm_parameters.keys():
            api_parameters['options'][p] = self.llm_parameters[p]

        response = requests.post(self.api_endpoint, json=api_parameters, stream=True)
        return response

    # Function for generating LLaMA2 llm_response
    def request_run_llm_llama2(self, context):
        """Run the Llama 2 LLM with the provided context, including the prompt as last message, in a suitable dialogue format"""

        dialogue = CTX_TEMPLATE + "\\n\\n"
        for message in context:
            if MSG_FORMAT in message.keys() and ROLE in message.keys():
                if message[MSG_FORMAT] == MSG_FORMAT_TXT:
                    role = message[ROLE]
                    if role == ROLE_US or role == ROLE_AS:
                        # include user and assistant messages in the prompt, 
                        # including generated code, not including interpreter output
                        dialogue += role + ": " + message[MSG] + "\\n\\n"
                    #elif role == ROLE_INT:
                        # do not include interpreter output

        items = self.run_llm_replicate(self.selected_llm_api_id, dialogue)
        item_function = lambda item: item
        #result = [f"Echo: {prompt}"]
        return (items, item_function)

    # Function for generating llm_response using replicate models without specific functions
    def request_run_llm_replicate(self, context):
        """Run a Replicate LLM with the provided context, including the prompt as last message, in a suitable dialogue format"""

        dialogue = CTX_TEMPLATE + "\n\n"
        for message in context:
            if MSG_FORMAT in message.keys() and ROLE in message.keys():
                if message[MSG_FORMAT] == MSG_FORMAT_TXT:
                    role = message[ROLE]
                    if role == ROLE_US or role == ROLE_AS:
                        # include user and assistant messages in the prompt, 
                        # including generated code, not including interpreter output
                        dialogue += role + ": " + message[MSG] + "\n\n"
                    #elif role == ROLE_INT:
                        # do not include interpreter output

        items = self.run_llm_replicate(self.selected_llm_api_id, dialogue)
        item_function = lambda item: item
        #result = [f"Echo: {prompt}"]
        return (items, item_function)

    def request_run_llm_ollama_parse_response(self, response):
        """Parses JSON responses from Ollama"""

        # Check if the response is complete or still ongoing
        response_last = response.decode('utf-8')
        if response_last.endswith('}\n') or response_last.endswith('}'):
            # response is complete
            response_text = ""
            for response_buffered in self.llm_response_buffer:
                response_text += response_buffered
            response_text += response_last

            self.llm_response_ongoing = False
            self.llm_response_buffer = []

            try:
                jsonr = json.loads(response_text)
                for k in ['prompt_eval_duration', 'eval_duration']:
                    if k in jsonr:
                        print("{}: {}".format(k, jsonr[k]))
                if 'context' in jsonr:
                    self.llm_returned_context = jsonr['context']
                if 'response' in jsonr:
                    return jsonr['response']
                if 'error' in jsonr:
                    return "LLM API returned error: {}".format(jsonr['error'])
            except json.decoder.JSONDecodeError:
                return ''

        elif not self.llm_response_ongoing and response_last.startswith('{'):
            # response is beginning
            self.llm_response_ongoing = True
            self.llm_response_buffer.append(response_last)
            return ''
        elif self.llm_response_ongoing:
            # response is ongoing
            self.llm_response_ongoing = True
            self.llm_response_buffer.append(response_last)
            return ''

        return ''


    # Function for generating an llm_response using Ollama
    def request_run_llm_ollama(self, prompt):
        """Run an LLM with Ollama with the provided context, including the prompt as last message, in a suitable dialogue format"""

        items = self.run_llm_rest_ollama(self.selected_llm_api_id, prompt)
        item_function = lambda item: self.request_run_llm_ollama_parse_response(item)

        #result = [f"Echo: {prompt}"]
        return (items, item_function)

    def request_run_llm_chatgpt4(self, context):
        """Run a ChatGPT LLM with the provided context, including the prompt as last message, in a suitable dialogue format"""

        dialogue = []
        for message in context:
            if MSG_FORMAT in message.keys() and ROLE in message.keys():
                if message[MSG_FORMAT] == MSG_FORMAT_TXT:
                    role = message[ROLE]
                    if role == ROLE_US or role == ROLE_AS:
                        dialogue.append({"role": message[ROLE], "content": message[MSG]})

        items = self.run_llm_openai(self.selected_llm_api_id, dialogue)
        item_function = lambda item: item.choices[0].delta.get("content", "")
        #result = [f"Echo: {prompt}"]
        return (items, item_function)

    def request_run_prompt(self, context, prompt):
        """Runs the provided prompt or context, which includes the prompt as last message """

        if self.selected_llm.startswith(API_REPLICATE + '/Llama2'):
            return self.request_run_llm_llama2(context)
        elif self.selected_llm.startswith(API_REPLICATE):
            return self.request_run_llm_replicate(context)
        elif self.selected_llm.startswith(API_OPENAPI):
            return self.request_run_llm_chatgpt4(context)
        elif self.selected_llm.startswith(API_OLLAMA):
            return self.request_run_llm_ollama(prompt)

    def clear_history(self):
        self.llm_returned_context = []
