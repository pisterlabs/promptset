from ottoai.templates import INSTRUCTION
from ottoai.helpers import llm_completion, create_string, extract_python_code_from_md, get_runner_function

import logging
import os
import json
import pkg_resources
import subprocess



class Assistant:
    """
    The Assistant class is responsible for managing the skills and conversations.
    """
    def __init__(self, name: str, personality: str, llm_engine, model: str, user_context_variables: dict = {}):
        """
        Initialize the assistant with a name, personality, language model engine, and model.
        """
        self.name = name
        self.personality = personality
        self.llm_engine = llm_engine
        self.model = model
        self.pip_skills = []
        self.user_context_variables = user_context_variables
    
    def _m(self, messages):
        return llm_completion(model=self.model, messages=messages)
        

    def set_user_context_variables(self, user_context_variables: dict = {}):
        """
        Set the user context variables for the assistant.
        
        Parameters:
        user_context_variables (dict): A dictionary containing the user context variables.
        """
        self.user_context_variables = user_context_variables

    def add_pip_skill(self, pip_module):
        """
        Add a new skill to the assistant.
        """

        installed_packages = pkg_resources.working_set
        installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])

        if pip_module not in installed_packages_list:
            try:
                installed_packages_pip_freeze = subprocess.check_output(["pip", "freeze"]).decode().split('\n')
                if pip_module not in (package.split('==')[0] for package in installed_packages_pip_freeze) and pip_module not in (package.split('==')[0] for package in installed_packages_pip_freeze):
                    raise ImportError(f"Trying to add skill, but pip module {pip_module} is not installed. \nTo solve this try: pip install {pip_module}")
            except subprocess.CalledProcessError:
                raise ImportError(f"Failed to execute pip freeze.")
        
        self.pip_skills.append(pip_module)
            
    
    def question(self, text: str):
        """
        Send a message to the assistant and return the assistant's response.
        """
        
        response = self.generate_and_run_code_for_question(text)
        return response
        

    def start_conversation(self, user_name: str):
        """
        Start a new conversation with the user.
        """
        return Conversation(self, user_name)


    def generate_and_run_code_for_question(self, question, retries_until_figured = 10):
        
        arg_data = {
            "context_variables": {key: "<...HIDDEN...>" for key in self.user_context_variables}
        }

        arg_data_all = {
            "context_variables": self.user_context_variables
        }
        
        
        arguments_dictionary_str =  create_string(arg_data)
        
        modules_metadata = ", ".join(self.pip_skills)

        instruction =INSTRUCTION.format(modules_metadata = modules_metadata, input_dictionary_string = arguments_dictionary_str, question=question)
        
        logging.debug("[OTTO] Generated Instruction: " + instruction)
        
        messages = [{"role": "system", "content": instruction}]
        error = 1
        code = ''
        error_message =  None
        for _ in range(retries_until_figured):
            
            if error_message:
                messages += [{"role": "system", "content":"ran {code} and had this error: {error}".format(code=code, error=error_message)}]
                
            logging.debug("[OTTO] Messages: \n" + json.dumps(messages, indent=4))
            resp = self._m(messages)
            code = resp['choices'][0]['message']['content']
            error_message = None
            try:
                function_code = extract_python_code_from_md(code)
                if function_code is not None:
                    code = function_code
            
                logging.debug("[OTTO] Generated Code: \n```\n{code}\n\n```\n".format(code = code))
                runner_function = get_runner_function(code)

                result = runner_function(input=arg_data_all)
                break
            except Exception as e:
                error_message = str(e)
                logging.debug("[OTTO] Exception occurred: " + str(e))
                logging.debug("[OTTO] Trying again: " + str(error))
                error += 1
                result = None
                continue
        
        try:
            logging.debug("[OTTO] Result: " + str(result))
        except:
            logging.debug("[OTTO] Result ")

        return result

class Conversation:
    """
    The Conversation class is responsible for managing the conversation between the user and the assistant.
    """
    def __init__(self, assistant: Assistant, user_name: str):
        """
        Initialize the conversation with the assistant, the user's name and context variables.
        """
        self.assistant = assistant
        self.user_name = user_name

    def message(self, text: str):
        """
        Send a message to the assistant and return the assistant's response.
        """
        
        response = self.assistant.generate_and_run_code_for_question(text)
        return response

    



if __name__ == "__main__":

    import os
    import openai
    import logger
    logging.basicConfig(level=logging.DEBUG)
    import json

    # create an assistant
    eve = Assistant(name="eve", personality="Like Scarlett Johansson with John Oliver's wits", llm_engine=openai, model="gpt-4-1106-preview")

    # Make your agent capable of answering questions and do all things Github, by simply passing the sdk module
    # Remember. Just pass objects as skills, and Otto will figure out the rest. 
    eve.add_pip_skill(pip_module='PyGithub')
    eve.add_pip_skill(pip_module="pandas")

    # Start a conversation and add user specific context (variable names must be interpretable, Otto will take care of the rest)
    # In this case the assistant will need access token to github, this is so you can pass context dynamically (solving for multitenancy)
    eve.set_user_context_variables({"github_api_token":os.getenv("GITHUB_TOKEN")}) 

    # ask a question
    response = eve.question("Who were the last 5 people to star the mindsdb/otto repo?")
    print(json.dumps(response, indent=4))

    # ask a question
    response = eve.question("What are the top 5 repos in ai?")
    print(json.dumps(response, indent=4))



