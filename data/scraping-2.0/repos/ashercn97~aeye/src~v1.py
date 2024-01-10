from pydantic import BaseModel, Field
import re
from .oai_funcs import *
import openai
import json


prompts = SystemPrompts()


# base class, simply adds the functionality to get the dictionary of all the attirbutes set, makes life easier
class AshBaseClass:
    # returns a dictionary of all the attributes of the class
    def dict(self):
        return {name: getattr(self, name) for name in self.__class__.__dict__ if not name.startswith('_') and not callable(getattr(self, name))}


'''
Main class

* takes a query as input, such as 2+2, and then you also set the following attirbutes:
lengths: a dictionary of the keyword and then the length/what it should be. This could be like "one name" or "one integer spelled out"
template: a string where the keyword is put between <> so for instance "The answer is <keyword>"
'''
class TextTemplate(AshBaseClass):
    # parameters that classes will have
    lengths = {} # will be in the form {"name": "length"} where length is something like " One Word " or " One Sentence "
    template = "" # will be in the form "hi <name> "

    def __init__(self, query):
        self.sys_prompt = """Below you are given a template. In the template there will be items like <name: One name>. In a situation like this, you will return a JSON where you have name: "Asher" for instance. \n\n"""
        self.us_prompt = query
        self.info = self.dict()
        self.__flatten__()

        # call all functions later...

    def run(self):
        self.final = self.__put_into_template__(self.__get_response__())
        return self.final

    # returns a string where the answered values are in between <> and the rest is the template, and then a list of the answered values between
    def __get_response__(self):
        client = setup_client()
        response = get_completion(client, self.us_prompt, self.sys_prompt, json=True).content
        response = json.loads(response)
        return response
    
    # reutnrs the original template but with the new values in it
    def __put_into_template__(self, response):
        new_template = self.template
        for key, val in response.items():
            new_template = self.__find_and_put__(key, val, new_template)
        return new_template


    #  "flatens" the template and lengths/types into a sytem prompt
    def __flatten__(self):
        values = self.__find_values__()
        for val in values:
            # automatically sets the system prompt 
            self.__sub_in__(val, f"<{val}: Choose a value that fits this description: {self.info["lengths"][val]} > \n\n")

        return self.sys_prompt

    # finds the different values in the template
    def __find_values__(self):
        # finds all the values in the template
        values = re.findall("<(.+?)>", self.template)
        return values
    

    # substitutes in the values for the template
    def __sub_in__(self, name, value):
        # replaces <name> with value in the template
        self.sys_prompt  += self.template.replace("<" + name + ">", str(value))

    def __find_and_put__(self, name, value, new_template):
        # replaces <name> with value in the template
        new_template = new_template.replace("<" + name + ">", str(value))
        return new_template

    # "type" checks the value
    def __type_check__(self, type, value):
        sys_prompt = prompts.TYPE_CHECKER
        us_prompt = UserPrompts.TYPE_CHECKER.format(type, value)
        client = setup_client()
        response = get_completion(client, us_prompt, sys_prompt)

        response = bool(response)

        if response:
            return True
        elif response == False:
            return False
        else:
            raise ValueError("chatGPT returned something wrong")
        



