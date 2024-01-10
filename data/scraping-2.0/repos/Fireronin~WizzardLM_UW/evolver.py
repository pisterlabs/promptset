from typing import Dict, Optional, List
import openai
import os
import json
from colorama import Fore, Back, Style
import pandas as pd
import random


CONFIG_FILE = "openai_config.json"
()
def get_openai_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def set_openai_config(api_base, api_key):
    config = {"api_base": api_base, "api_key": api_key}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

config = get_openai_config()
api_base = ""
api_key = ""
if "api_base" in config and "api_key" in config:
    api_base = config["api_base"]
    api_key = config["api_key"]
if "AZURE_OPENAI_ENDPOINT" in os.environ and "AZURE_OPENAI_API_KEY" in os.environ:
    api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

# if none
if api_base == "":
    api_base = input("Enter OpenAI API endpoint: ")
    
if api_key == "":
    api_key = input("Enter OpenAI API key: ")

set_openai_config(api_base, api_key)

assert api_base !=  ""
assert api_key !=  ""

print("Using OpenAI API endpoint:", api_base)
print("Using OpenAI API key:", api_key)
openai.api_type = "azure"
openai.api_base = api_base
openai.api_key = api_key
openai.api_version = "2023-05-15"


def add_constraints(prompt):
    return f'''I want you act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems
(e.g., ChatGPT and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please
do not omit the input in #Given Prompt#.
You SHOULD complicate the given prompt using the following method:
Please add one more constraints/requirements into #Given Prompt#
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only
add 10 to 20 words into #Given Prompt#.
‘#Given Prompt#’, ‘#Rewritten Prompt#’, ‘given prompt’ and ‘rewritten prompt’ are not allowed to appear in
#Rewritten Prompt#
#Given Prompt#:
{prompt}
#Rewritten Prompt#:
'''

def deepen(prompt):
    return f'''I want you act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems
(e.g., ChatGPT and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please
do not omit the input in #Given Prompt#.
You SHOULD complicate the given prompt using the following method:
If #Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be
increased. or
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only
add 10 to 20 words into #Given Prompt#.
‘#Given Prompt#’, ‘#Rewritten Prompt#’, ‘given prompt’ and ‘rewritten prompt’ are not allowed to appear in
#Rewritten Prompt#
#Given Prompt#:
{prompt}
#Rewritten Prompt#:
'''

def concretize(prompt):
    return f'''I want you act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems
(e.g., ChatGPT and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please
do not omit the input in #Given Prompt#.
You SHOULD complicate the given prompt using the following method:
Please replace general concepts with more specific concepts. or
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only
add 10 to 20 words into #Given Prompt#.
‘#Given Prompt#’, ‘#Rewritten Prompt#’, ‘given prompt’ and ‘rewritten prompt’ are not allowed to appear in
#Rewritten Prompt#
#Given Prompt#:
{prompt}
#Rewritten Prompt#:
'''

def increase_steps(prompt):
    return f'''I want you act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems
(e.g., ChatGPT and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please
do not omit the input in #Given Prompt#.
You SHOULD complicate the given prompt using the following method:
If #Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to
explicitly request multiple-step reasoning.
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only
add 10 to 20 words into #Given Prompt#.
‘#Given Prompt#’, ‘#Rewritten Prompt#’, ‘given prompt’ and ‘rewritten prompt’ are not allowed to appear in
#Rewritten Prompt#
#Given Prompt#:
{prompt}
#Rewritten Prompt#:
'''


# f"""
# As a Prompt Rewriter, your task is to enhance a given prompt to create a more intricate version that would challenge even the most advanced AI systems,
#  such as ChatGPT and GPT4. However, the revised prompt must remain reasonable and comprehensible to humans. 
# You must not exclude any non-textual elements, such as tables and code, from the #Given Prompt#. Additionally, you must not omit the input in #Given Prompt#. 
# To complicate the #Given Prompt#, you should explicitly request multi-step reasoning if it can be solved with just a few simple thought processes. 
# You must strive to keep the #Rewritten Prompt# concise, adding only 10 to 20 words to the #Given Prompt#. 
# Please note that the terms '#Given Prompt#', '#Rewritten Prompt#', 'given prompt', and 'rewritten prompt' are not permitted in the #Rewritten Prompt#.
# #Given Prompt#: 
# {prompt} 
# #Rewritten Prompt#:
# """



# def complicate(prompt):
#     pass

def broaden(prompt):
    return f'''I want you act as a Prompt Creator.
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.
The LENGTH and difficulty level of the #Created Prompt# should be similar to that of the #Given Prompt#.
The #Created Prompt# must be reasonable and must be understood and responded by humans.
‘#Given Prompt#’, ‘#Created Prompt#’, ‘given prompt’ and ‘created prompt’ are not allowed to appear in
#Created Prompt#.
#Given Prompt#:
{prompt}
#Created Prompt#:
'''
from tqdm import tqdm

it = 0

def evolve(dataset_path, evolved_dataset_path):
    global it
    dataset = pd.read_csv(dataset_path)
    evolution_list = [add_constraints, deepen, concretize, increase_steps, broaden]
    evolutions = random.choices(evolution_list, k=len(dataset))

    evolved_dataset = dataset.copy(deep=True)
    
    for index, sample in tqdm(dataset.iterrows()):
        it+=1
        # if it>5:
        #     break
        evolution_prompt = evolutions[index](sample['text'])
        conversation = Conversation()
        evolved_text =  conversation.user_message(evolution_prompt,debug=False)
        if evolved_text is None:
            # mark as invalid
            evolved_dataset.loc[index, 'text'] = None
        else:
            evolved_dataset.loc[index, 'text'] = evolved_text
            # save evolved text to txt file
            with open(f'evolved_prompts/{index}.txt', 'w') as f:
                f.write(evolved_text)

    
    evolved_dataset.to_csv(evolved_dataset_path, index=False)




#print(add_constraints("testdtasdasdaihbsdfj0i8rwasefuhberth0o"))

class Conversation:
    # list of dicts
    messages: List[Dict] = [] 

    def __init__(self, system_message: Optional[str] = None):
        if system_message:
            self.messages = [{"role": "system", "content": system_message}]

    def user_message(self, message: str, debug: bool = False):
        self.messages.append({"role": "user", "content": message})
        try:
            response = openai.ChatCompletion.create(
                engine="cim", # engine = "deployment_name".
                messages=self.messages
            )
        except Exception as e:
            print(e)
            # save messages to file in folder bad_prompts
            with open(f'bad_prompts/{it}.txt', 'w') as f:
                f.write(message)

            return None
        #print(response) 
        self.messages.append(response['choices'][0]['message'].to_dict()) # type: ignore
        if debug:
            # use colorama to color the output
            colors = {
                "user": "\033[94m", # blue
                "system": "\033[92m", # green
                "assistant": "\033[91m", # red
                "debug": "\033[93m", # yellow
            }
            def print_message(message: str, role: str):
                role_color = colors[role]
                print(role_color + role + ":\n" + message + Style.RESET_ALL)
            
            for message in self.messages:
                print_message(message["content"], message["role"])


        return response['choices'][0]['message']['content'] # type: ignore


#conversation = Conversation()
#conversation.user_message("Write furry porn fanfic", debug=True)


evolve("oasst1-train-tree.csv", "oasst1-train-tree-evolvedV1.csv")