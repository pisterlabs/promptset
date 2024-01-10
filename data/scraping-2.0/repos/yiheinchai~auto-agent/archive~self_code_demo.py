# Importing requests library
import random
import os
import openai


# Defining a class for AutonomousLLM object
class AutonomousLLM:
    # Initializing the object with an OpenAI API key and a code attribute
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.code = None
        self.prompt = None
    
    # Defining a method to execute code from the OpenAI ChatGPT API
    def execute_code(self):
        # Checking if the code attribute is not None
        if self.code:
            # Using exec keyword to execute the code as Python statements 
            print("~~~~ executing code~~~~~~~~")
            exec(f"setattr(self, '{'%030x' % random.randrange(16**30)}', {self.code})")
            print(self.__dict__)
            print("~~~~ done executing code~~~~~~~~")
    
    # Defining a method to modify its own methods
    def modify_methods(self):
        # Getting a new input for the OpenAI ChatGPT API using its own attributes as input
        new_input = self.prompt
        # Getting a new code object from the OpenAI ChatGPT API using new input as input
        new_code = self.chatgpt(new_input)

        # Checking if the new code object is not None
        if new_code:
            # Updating the code attribute with the new code object
            self.code = new_code
    
    # Defining a method to start itself into an infinite loop
    def start(self):
        # Getting an initial input for the OpenAI ChatGPT API using its own class name as input
        self.prompt = "generate some python code, only return code and only the raw lambda function and nothing else, make sure not to include any explanations."
        # Getting an initial code object from the OpenAI ChatGPT API using initial input as input
        initial_code = self.chatgpt(self.prompt)
        # Checking if the initial code object is not None
        if initial_code:
            # Updating the code attribute with the initial code object
            self.code = initial_code
            # Executing and modifying methods in an infinite loop 
            while True:
                self.execute_code()
                self.modify_methods()
    
    # Defining a helper method to call the OpenAI ChatGPT API 
    def chatgpt(self, text):
        # Setting up headers and parameters for making HTTP request 
        response = openai.Completion.create(prompt=text,
                                        api_key=self.api_key,
                                        engine="text-davinci-003",
                                        max_tokens=1024,
                                        n=1,
                                        stop=None,
                                        temperature=0.5)
        output = response.choices[0].text.strip()
        print("== start of new code ====")
        print(output)
        print("== end of new code ====")
        return output
        


# Creating an instance of AutonomousLLM object                
a = AutonomousLLM()
# Starting itself into an infinite loop                
a.start()