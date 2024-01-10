import openai
from utils import is_dialogue_format, extract_dialogues
import json
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

def get_character_dialogues():
    output = {
        'Characters': Characters,
        'Dialouges': Dialogues
    }

# openai.api_key = os.getenv("OPENAI_API_KEY")

functions = [
    {
        'name': 'get_character_dialogue',
        'description': 'Given a script, retrieves the characters and their respective dialogues in a sequential manner',
        'parameters': {
            'type':'object',
            'properties': {
                'Characters':{
                    'type':'string',
                    'description': 'List all the characters in the script in a comma separated list',
                },

                'Dialogues':{
                    'type':'object',
                    'properties':{
                        'Character':{
                            'type': 'string',
                            'description': 'Name of character speaking the dialogue.'
                        },
                        'Dialogue':{
                            'type':'string',
                            'description':'Dialogue spoken by current character.'
                        }
                    },
                    # 'description':'List of dialogues in the script, along with associated character'
                },
            },
            'required':['Characters', 'Dialogues']

        }
    }
]

PROMPT = '''You are a magnificent author and an avid book reader. Due to your vase knowledge and wisdom you have the capability to identify all the characters in a scene with utmost accuracy. I will provide you with scripts - they may be lines, paragraphs or pages. You must analyze the entire text block and realize, keeping in mind the previous conversations, who should be narrating the line - Is it a narrator, an already seen character (mention his/her name) or  a new character (mention the new character(s) with their names)?

next, convert the script into a set of dialogues.
char1: line1
char2: line2
...

All instances similar to "he said" or any of the "narrator-specific" lines should only go to the narrator. Don't miss any words.

Always reply in this format:
```
Characters: character1, character2 and so on

Script:
character1: dialogue1
character2: dialogue2
```

Example1:
"I want apples," said Polymars. He was in a rush to the grocery store. "Bring some for me as well," demanded Mr. Braun.

Output:
```
Characters: Narrator, Polymars, Mr. Braun

Script:
Polymars: I want apples
Narrator: said Polymars. He was in a rush to the grocery store.
Mr. Braun: Bring some for me as well
Narrator: demanded Mr.Braun.
```
'''

def extract_story(script, key):
    user_message = {
        'role': 'user',
        'content': PROMPT + "\n\nGive me the character and Dialogues. Here is your script: \n```\n" + script + "\n```"
    }
    openai.api_key = key
    # Query the model
    response = openai.ChatCompletion.create(
        model = "gpt-4",
        # model = 'gpt-3.5-turbo',
        messages = [ user_message ],
        functions = functions,
        function_call = {'name' : 'get_character_dialogue'}
    )
    # print(response['choices'][0]['message']['function_call']['arguments'])
    # print(type(response['choices'][0]['message']['function_call']['arguments']))
    response = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    # print(response['Characters'])
    # print(response['Dialogues'])
    print(response)
    return response
