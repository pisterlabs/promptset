import openai
import sys 
import os
import json 
from settings import OPENAI_API_KEY
import random
from typing import Tuple 

def test_openai_call( msg="Hello, write me a story. Two paragraphs", api_key = OPENAI_API_KEY) -> openai.ChatCompletion:
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content": msg}],
        stream=True
    )
    return response 
def save_response(response, folder="./testing/example_streaming/"):
    os.makedirs(folder, exist_ok=True)
    numb = len(os.listdir(folder))
    with open(f"{folder}example{numb}.json", "w") as f:
        json.dump(response, f, indent=4)
def handle_stream(response) -> Tuple[str, list]:
    collected = []
    txt  = ''
    for event in response:
        collected.append(event)
        if event.choices[0].finish_reason is not None:
            print("\n")
            print("Done! Got a finish reason")
            break
        letter = event.choices[0].delta.content
        sys.stdout.write(letter)
        sys.stdout.flush()
        txt += letter
    return txt, collected    

response = test_openai_call(msg="Write the following two words for testing purposes: 'Hello World!' Nothing else please, I am testing something out. ")

print(response)
print(dir(response))
# collected = []
# completion_txt = ""
# for event in response:
#     #print(event)
#     collected.append(event)
#     try:
#         text = event.choices[0].delta.content
#         completion_txt += text
#         sys.stdout.write(text)
#         sys.stdout.flush()
#     except KeyError:
#         print()
#         print("Got Nothing")  
#     except AttributeError:
#         print()
#         print("Attribute Error")  

# print("Done!")
txt, collected = handle_stream(response)

#save_response(collected)
   





    
    