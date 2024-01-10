import openai
import json
from .handlers import FunctionHandler



class ChatSession:

    def __init__(self, api_key, model="gpt-3.5-turbo-0613", system_message=None, example_path=None, function_file_paths=None, overwrite=False):
        openai.api_key = api_key
        self.model = model
        self.messages = [{"role": "system", "content": system_message}] if system_message else []
        self.example_path = example_path
        self.function_file_paths = function_file_paths
        self.function_call = {}
        
        if function_file_paths:
            handler = FunctionHandler(api_key, function_file_paths, example_path, overwrite=overwrite)
            self.functions, self.function_refs = handler.create_function_list_and_refs()
        else:
            self.functions = []
            self.function_refs = {}


    def add_function(self, name, description, parameters, ref=None):
        self.functions.append({
            "name": name,
            "description": description,
            "parameters": parameters
        })
        if ref:
            self.function_refs[name] = ref


    def send_message(self, user_message, stream=False):
        self.messages.append({"role": "user", "content": user_message})
        if self.functions:
            params = {
                "model": self.model,
                "messages": self.messages,
                "functions": self.functions,
                "stream": stream,
            }
        else:
            params = {
                "model": self.model,
                "messages": self.messages,
                "stream": stream,
            }

        response = openai.ChatCompletion.create(**params)
        if not stream:
            return self.handle_normal_response(response)
        return response

    
    def handle_normal_response(self, response): # Just returns entire response in text
        message  = response['choices'][0]['message']
        if message.get('function_call'):

            self.messages.append(message)
            # Return as dict
            resp = list(response.choices)[0]
            return resp.to_dict()["message"]["function_call"]


            # function_name = message['function_call'].get('name', None)
            # function_args = message['function_call'].get('arguments', None)
            # if function_args:
            #     function_args = json.loads(function_args)
            #     function_response = str(self.call_function(function_args, function_name))
            #     # Appending both message and function response to messages
            #     self.messages.append(message)
            #     # self.messages.append({"role": "function", "content": function_response, "name": function_name})
            #     return message['function_call']
        else:
            self.messages.append(message)
            return message['content'] 
        
    
    def stream_response(self, response, function_for_output):
        func_call = {
            "name": None,
            "arguments": "",
        }
        function_called = False
        whole_content = ""

        for chunk in response:
            delta = chunk.choices[0].delta
            if "function_call" in delta:
                function_called = True
                new_func_call = self.handle_function_call_chunk(chunk)
                if not func_call["name"]:
                    func_call["name"] = new_func_call["name"]
                if "arguments" in new_func_call:
                    func_call["arguments"] += new_func_call["arguments"]
            else:
                content = self.handle_normal_chunk(chunk)
                whole_content += content
                function_for_output(content)


        self.messages.append({"role": "assistant", "content": whole_content})
        # Call function
        if function_called:
            args = json.loads(func_call["arguments"])
            function_response = self.call_function(args, func_call["name"])
            self.messages.append({"role": "function", "content": function_response, "name": func_call["name"]})
            function_for_output(function_response)

    def handle_function_call_chunk(self, chunk):
        func_call = {
            "name": None,
            "arguments": "",
        }
        delta = chunk.choices[0].delta
        if "name" in delta["function_call"]:
            func_call["name"] = delta["function_call"]["name"]
        if "arguments" in delta["function_call"]:
            func_call["arguments"] = delta["function_call"]["arguments"]
        # if chunk.choices[0].finish_reason == "stop" or chunk.choices[0].finish_reason == "function_call":
        #     complete = True

        return func_call
    

    def handle_normal_chunk(self, chunk):
        if "content" in chunk.choices[0].delta:
            return chunk.choices[0].delta.content
        return ""


    def call_function(self, arguments, name, safe_mode=True):
        if safe_mode:
            # console.print(f"\nAttempting to call function [bold]{name}[/bold] with arguments [bold]{arguments}[/bold]. y/n?", style="bold red")
            response = input("\nYou: ")
            if response.lower() != "y":
                return "User cancelled function call."



        ref = self.function_refs.get(name, None)
        if ref:
            # console.print(f"\nCalling function [bold]{name}[/bold] with arguments [bold]{arguments}[/bold]...", style="magenta")
            if arguments:
                return ref(**arguments)
            return ref()
        
    def typewriter_print(self, stream_response, speed=0.001):
        from time import sleep
        import sys 
        for char in stream_response:
            sleep(speed)
            sys.stdout.write(char)
            sys.stdout.flush()





