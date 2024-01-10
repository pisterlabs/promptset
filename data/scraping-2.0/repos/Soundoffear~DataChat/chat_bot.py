import openai
import json

openai.api_key = "<YOUR OPENAI API KEY>"

class ChatBot:
    
    def __init__(self, system="", filename="messages.json", max_messages=10):
        self.system = system
        self.messages = []
        self.filename = filename
        self.max_messages = max_messages
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        
        if self.system:
            self.messages.append({"role":"system", "content":system})
            
    def __call__(self, message, functions=None):
        self.messages.append({"role":"user", 
                              "content":message})
        result = self.execute(functions=functions)
        print("RESULT", result)
        result_save = result["choices"][0]["message"]["content"]
        print("RESULT SAVE", result_save)
        if result_save == None and result["choices"][0]["message"]["function_call"]["name"] != "data_analysis":
            result_save = f"Calling Function and executing: {message}"
        else:
            if result["choices"][0]["message"].get("function_call") is not None:
                msg = result["choices"][0]["message"]["function_call"]["arguments"]
                result_save = f"Calling Function and executing: {msg}"
            else:
                result_save = f"Calling Function and executing: {result_save}"
        self.messages.append({"role":"assistant", "content":result_save})

        with open(self.filename, "a") as f:
            f.write(json.dumps(self.messages[-2:]) + "\n")

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
            
        return result
        
    def execute(self, functions=None):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", 
                                                  messages=self.messages,
                                                  functions=functions,
                                                  function_call="auto")
        print("COMPLETION", completion)
        count = 0

        while True:
        
            content = completion["choices"][0]["message"]["content"]
            
            if completion["choices"][0]["message"].get("function_call") is not None:

                if completion["choices"][0]["message"]["function_call"]["name"] == "db_questions":
                    return completion

                if content == None or completion["choices"][0]["message"]["function_call"]["name"] == "data_analysis":
                    count = 0
                    break
                
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", 
                                                    messages=self.messages,
                                                    functions=functions,
                                                    function_call="auto")
            else:
                break
            
            if count > 5:
                count = 0
                break
            print(count)
            count += 1
            
        for msg in self.messages:
            print(msg)
            
        return completion

    def get_all_messages(self):
        return self.messages

    def get_message_count(self):
        return len(self.messages)

    def get_token_usage(self):
        return {
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens
        }