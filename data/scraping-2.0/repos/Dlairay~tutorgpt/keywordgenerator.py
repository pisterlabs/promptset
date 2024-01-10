import os
from dotenv import load_dotenv
from openai import OpenAI
import ast

load_dotenv()
api_key= os.getenv("OPENAI_APIKEY")
client = OpenAI(api_key=api_key)

# string1 ="""• Apply both the Energy and Momentum Conservation Principle to a system of objects to determine the velocities of objects after an interaction.
# • Apply the Momentum Conservation Principle to a system undergoing recoil.
# • Solve 1D problems in which two bodies collide with each other
# • Distinguish among elastic, super-elastic, inelastic, and completely inelastic collisions"""

# list_a = string1.split("•")
# if list_a[0] == '':
#     list_a.remove('')
# list_a = [string[:-1] for string in list_a]
# print(list_a)

def generate_keyword_list(list_a):
    prompt = "here is a list of learning objectives{}, you will iterate through the list and simplify the sentence into its core idea, and provide key words that i can search for in youtube.If there are duplicates remove them. you will then stricly output a python list containing those key words do not add courtesy messages".format(list_a)

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "you are helpful study tutor for a STEM university student"},
        {"role": "user", "content": prompt}
    ]
    )

    chatgpt_output = completion.choices[0].message.content
    converted_list = ast.literal_eval(chatgpt_output)
    return converted_list

   







