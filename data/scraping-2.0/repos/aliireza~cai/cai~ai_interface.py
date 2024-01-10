import abc
import bardapi
import openai
from EdgeGPT import Query, Cookie
from colorama import Fore, Style
import os
import sys


# Abstract base class for all AI interfaces
class AIInterface(abc.ABC):
    def __init__(self, version=0):
        # Initialize AI with specific configurations
        self.init(version)

    @abc.abstractmethod
    def init(self, version):
        # Each AI has its own specific initialization
        pass

    @abc.abstractmethod
    def submit_task(self, task, code=None, role=""):
        # Submit a task to the AI service and return the response
        pass

    @abc.abstractmethod
    def get_code(self, response):
        # Parse the response from the AI service to extract code
        pass


class Bard(AIInterface):
    def init(self, version):
        self.type = "BARD"
        # Configure the BARD API key
        # os.environ['_BARD_API_KEY'] = "placeholder"
        bardapi.api_key = os.environ.get('_BARD_API_KEY', 'Not Set')
        self.bard = bardapi.core.Bard()

    def submit_task(self, task, code=None, role=""):
        # Submit the task to BARD and handle the response
        prompt = task
        if code is not None:
            prompt = prompt + "\n" + code
        response = self.bard.get_answer(prompt)
        if ('Error' in response.get('content')):
            print(Fore.RED + "Error in BARD API." + Style.RESET_ALL)
            sys.exit(1)
        if code is not None:
            return self.get_code(response.get('content'))
        else:
            return response.get('content')

    def get_code(self, response):
        # Parse the response from BARD to extract code
        index = response.find("```")
        code = response[index:].strip()
        index = code.find("c++")
        code = code[index+3:].strip()
        index = code.find("```")
        code = code[:index].strip()
        return code


class GPT(AIInterface):
    def init(self, version):
        self.type = "GPT"
        self.version = version
        # Configure the OpenAI GPT API key
        # os.environ['OPENAI_API_KEY'] = "placeholder"
        openai.api_key = os.environ.get('OPENAI_API_KEY', 'Not Set')
        self.gpt = openai
        if(version == 3.5):
            self.model = "gpt-3.5-turbo"
        elif(version == 4):
            self.model = "gpt-4"
        '''A temperature of 0 means the responses will be very straightforward, almost deterministic (meaning you almost always get the same response to a given prompt)
        A temperature of 1 means the responses can vary wildly.'''
        self.temperature = 0
        #self.submit_task("Always give me only code with syntax highlighting; for example ```c++ int main()```", "", "system")
        self.submit_task("return only code; nothing else","","system")

    def submit_task(self, task, code=None, role="user"):
        # Submit the task to GPT and handle the response
        prompt = task
        if code is not None:
            prompt = prompt + "\n" + code
        messages = [{"role": role, "content": prompt}]
        response = self.gpt.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,)
        if ('Error' in response):
            print(Fore.RED + "Error in OpenAI API." + Style.RESET_ALL)
            sys.exit(1)
        if code is not None:
            return self.get_code(response.choices[0].message["content"])
        else:
            return response.choices[0].message["content"]

    def get_code(self, response):
        # Parse the response from GPT to extract code
        index = response.find("```")
        code = response[index:].strip()
        index = code.find("cpp")
        code = code[index+3:].strip()
        index = code.find("```")
        code = code[:index].strip()
        return code


class Bing(AIInterface):
    def init(self, version):
        self.type = "BING"
        # Configure the Bing API
        os.environ['BING_U'] = ""
        self.bing = Query
        self.style = "precise"  # creative, balanced, or precise
        self.content_type = "text"  # "text" for Bing Chat; "image" for Dall-e
        self.cookie_file = "./bing_cookies_1.json"
        c = Cookie()
        c.current_filepath = self.cookie_file
        c.import_data()
        # echo - Print something to confirm request made
        # echo_prompt - Print confirmation of the evaluated prompt

    def submit_task(self, task, code=None, role=""):
        # Submit the task to Bing and handle the response
        prompt = task
        if code is not None:
            prompt = prompt + "\n" + code
        response = self.bing(prompt, self.style, self.content_type,
                             0, echo=False, echo_prompt=False)
        if ('Error' in response.output):
            print(Fore.RED + "Error in Bing API." + Style.RESET_ALL)
            sys.exit(1)
        if code is not None:
            return self.get_code(response)
        else:
            return response.output

    def get_code(self, response):
        # Parse the response from Bing to extract code
        return response.code


# Function to build the AI interface based on the user's choice
def AIBuilder(ai_choice):
    if ai_choice == 'BARD':
        return Bard()
    elif ai_choice == 'BING':
        return Bing()
    elif ai_choice == 'GPT-3.5':
        return GPT(3.5)
    elif ai_choice == 'GPT-4':
        return GPT(4)
