import openai
import os
from dotenv import load_dotenv

load_dotenv()

auth_key = os.getenv('auth_api_key')
openai.api_key= auth_key

with open('static/prompt.txt', 'r') as f:
    prompt = f.read()

# input prompt
print('Enter number of days: ')
days = input()
print('Enter vacation destination: ')
destination = input()

input_prompt = ('List '+ days + 'days itinerary at' + destination)

# Set the model and prompt
model_engine = 'text-davinci-003'
compiled_prompt = prompt + input_prompt

def text_completion():
# Generate a response
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=compiled_prompt,
        max_tokens=500,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0, 
        stop = ['\#']
    )
    
    result = completion.choices[0].text
    return result


def textfile_separator():
    
    output = text_completion()
    output_formatted = str(output.replace('Day', '++Day'))

    split_output = output_formatted.split('++')

    for i in split_output:
        split_days = i.split('\n\n')
        
        for i in split_days:
            print(i)
        
    
textfile_separator()
