from application.llm_handler import LlmHandler
from pika import BasicProperties
import openai
import logging
import tiktoken
import time

logger = logging.getLogger(__name__)

class OpenAIChatApi(LlmHandler):
    def __init__(self):        
        super().__init__()

    def validate(self, request):        
        is_valid, errors = self.validate_request(request, 'llm')
        return is_valid, errors

    def get_token_count(self, input_text):
        enc = self.token_counter.encode(input_text)
        return len(enc)
    
    def update_config(self, config_data):
        current_config = self.model_config
        merged_config = {**current_config, **config_data}
        self.model_config = merged_config

    def clip_messages(self, request, config):        
        clipped_messages = []
        messages, system_prompt_tokens, request_system_message, system_prompt, sys_prompt_in_request, max_input_tokens = self._prep_prompt(request, config)       
        input_token_count = system_prompt_tokens

        for index, message in enumerate(messages):
            token_count = self.get_token_count(message["content"]) 
            if token_count + input_token_count > max_input_tokens:
                break

            input_token_count += token_count
            clipped_messages.append(message)
        
        clipped_messages = clipped_messages[::-1]          
        if sys_prompt_in_request:
            clipped_messages.insert(0, request_system_message)

        return clipped_messages, input_token_count
    
    def execute(self, model, request):
        
        self.token_counter = tiktoken.encoding_for_model(model["model_name"])    
        clipped_messages, input_token_count = self.clip_messages(request, self.model_config)
        if clipped_messages == None:
            return None
        
        max_new_tokens, top_p, top_k, seed, temperature, stream_output, debug, stop_key, \
            min_p, mirostat, mirostat_eta, mirostat_tau = self.load_config_settings(input_token_count, request)
        if debug:
            print('\033[94m')
            print(request)
            print('\033[0m')                                     

        # make API request to OpenAI
        begin_time = time.time()        
        config = self.model_config

        check_stop_token, stop_conditions = self.build_stop_conditions(config["stop_on"])        
        response = openai.ChatCompletion.create(
          model=model["model_name"],
          stream=stream_output,
          messages=clipped_messages,
          temperature=temperature,
          max_tokens=max_new_tokens,
          stop=stop_conditions,
          presence_penalty=config.get("presence_penalty", 0),
          frequency_penalty=config.get("frequency_penalty", 0),
          top_p=top_p
        )

        channel = model["amqp_channel"]
        incoming_headers = model["amqp_headers"]

        # copy amqp headers
        outgoing_headers = {}
        for incoming_header in incoming_headers:
            if incoming_header in ["x-delay", "return_exchange", "return_routing_key"]:
                continue
            outgoing_headers[incoming_header] = incoming_headers[incoming_header]        

        response_str = ""
        finish_reason = "stop"                
        new_tokens = 0
        if stream_output:
            socket_id = incoming_headers["socket_id"] if "socket_id" in incoming_headers else None
            outgoing_headers["command"] = "prompt_fragment" if "stream_to_override" not in incoming_headers else incoming_headers["stream_to_override"]
            outgoing_properties = BasicProperties(headers=outgoing_headers)
            stop_generation_counter = 0

            for chunk in response:                
                stop_generation, stop_generation_counter = self.check_stop_generation(stop_generation_counter, 
                                    model["stop_generation_event"], model["stop_generation_filter"], socket_id)
                
                if stop_generation:
                    finish_reason = "abort"
                    break                

                new_tokens += 1
                if "content" in chunk['choices'][0]['delta']:                
                    if len(chunk['choices'][0]['delta']['content']) == 0:
                        continue

                    if debug:
                        print('\033[96m' + chunk['choices'][0]['delta']['content'], end="")

                    response_str += chunk['choices'][0]['delta']['content']
                    channel.basic_publish(
                        exchange=incoming_headers['return_exchange'], 
                        routing_key=incoming_headers['return_routing_key'], 
                        body=chunk['choices'][0]['delta']['content'], properties=outgoing_properties)

            if debug:
                print('\033[0m' + "")
        else:
            response_str = response['choices'][0]['message']['content']
            new_tokens = response['usage']['completion_tokens']

        end_time = time.time()
        elapsed = end_time - begin_time
        token_rate = 0 if elapsed == 0 else (new_tokens / elapsed)        
        model_name = incoming_headers["model_name"] if "model_name" in incoming_headers else "not_provided"
        resp = self.finish_response(stop_key, response_str, request, stream_output, finish_reason, 
                                        token_rate, new_tokens, input_token_count, model_name, elapsed, debug)        
        return resp

    def load(self, model, model_options, local_path):         
        self.model_config = model["configuration"]            
        openai.api_key = model["secrets"]["token"]
        return { "model_name": model["configuration"]["model"] }
